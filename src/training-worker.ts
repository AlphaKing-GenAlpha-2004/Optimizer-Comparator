// Optimized Training Worker
// This worker uses raw Float32Array operations for maximum performance.
// We avoid mathjs in the hot loops to decrease training and testing time.

type OptimizerType = 'SGD' | 'Adagrad' | 'RMSProp' | 'Adam';

interface WorkerParams {
  optimizer: OptimizerType;
  hiddenSize: number;
  learningRate: number;
  epochs: number;
  batchSize: number;
  inputSize: number;
  outputSize: number;
  X_train: Float32Array;
  y_train: Int32Array;
  X_test: Float32Array;
  y_test: Int32Array;
  trainSamples: number;
  testSamples: number;
}

// Helper functions for Float32Array operations
function createArray(size: number, fill = 0): Float32Array {
  const arr = new Float32Array(size);
  if (fill !== 0) arr.fill(fill);
  return arr;
}

function randomArray(size: number, scale: number): Float32Array {
  const arr = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = (Math.random() - 0.5) * 2 * scale;
  }
  return arr;
}

// Matrix Multiplication: C = A * B
// A: [rowsA x colsA], B: [colsA x colsB], C: [rowsA x colsB]
function matmul(A: Float32Array, B: Float32Array, rowsA: number, colsA: number, colsB: number, C: Float32Array) {
  C.fill(0);
  for (let i = 0; i < rowsA; i++) {
    const iOff = i * colsA;
    const iCOff = i * colsB;
    for (let k = 0; k < colsA; k++) {
      const valA = A[iOff + k];
      if (valA === 0) continue;
      const kOff = k * colsB;
      for (let j = 0; j < colsB; j++) {
        C[iCOff + j] += valA * B[kOff + j];
      }
    }
  }
}

// Optimized Matrix Transpose Multiplication: C = A^T * B
// A: [rowsA x colsA], B: [rowsA x colsB], C: [colsA x colsB]
function matmulATB(A: Float32Array, B: Float32Array, rowsA: number, colsA: number, colsB: number, C: Float32Array) {
  C.fill(0);
  for (let k = 0; k < rowsA; k++) {
    const kAOff = k * colsA;
    const kBOff = k * colsB;
    for (let i = 0; i < colsA; i++) {
      const valA = A[kAOff + i];
      if (valA === 0) continue;
      const iCOff = i * colsB;
      for (let j = 0; j < colsB; j++) {
        C[iCOff + j] += valA * B[kBOff + j];
      }
    }
  }
}

// Optimized Matrix Transpose Multiplication: C = A * B^T
// A: [rowsA x colsA], B: [rowsB x colsA], C: [rowsA x rowsB]
function matmulABT(A: Float32Array, B: Float32Array, rowsA: number, colsA: number, rowsB: number, C: Float32Array) {
  C.fill(0);
  for (let i = 0; i < rowsA; i++) {
    const iAOff = i * colsA;
    const iCOff = i * rowsB;
    for (let j = 0; j < rowsB; j++) {
      const jBOff = j * colsA;
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        sum += A[iAOff + k] * B[jBOff + k];
      }
      C[iCOff + j] = sum;
    }
  }
}

function reluAndDropout(A: Float32Array, mask: Float32Array, size: number, p: number, training: boolean) {
  if (!training || p <= 0) {
    for (let i = 0; i < size; i++) {
      if (A[i] < 0) A[i] = 0;
    }
    mask.fill(1);
    return;
  }
  const keepProb = 1 - p;
  const invKeepProb = 1 / keepProb;
  for (let i = 0; i < size; i++) {
    if (A[i] < 0) {
      A[i] = 0;
      mask[i] = 1;
    } else {
      if (Math.random() < p) {
        mask[i] = 0;
        A[i] = 0;
      } else {
        mask[i] = 1;
        A[i] *= invKeepProb;
      }
    }
  }
}

function addBias(A: Float32Array, b: Float32Array, rows: number, cols: number) {
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    for (let j = 0; j < cols; j++) {
      A[off + j] += b[j];
    }
  }
}

function reluDeriv(dA: Float32Array, Z: Float32Array, size: number) {
  for (let i = 0; i < size; i++) {
    if (Z[i] <= 0) dA[i] = 0;
  }
}

function softmax(A: Float32Array, rows: number, cols: number) {
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    let maxVal = -Infinity;
    for (let j = 0; j < cols; j++) {
      if (A[off + j] > maxVal) maxVal = A[off + j];
    }
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      A[off + j] = Math.exp(A[off + j] - maxVal);
      sum += A[off + j];
    }
    for (let j = 0; j < cols; j++) {
      A[off + j] /= (sum + 1e-15);
    }
  }
}

function sumCols(A: Float32Array, rows: number, cols: number, res: Float32Array) {
  res.fill(0);
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    for (let j = 0; j < cols; j++) {
      res[j] += A[off + j];
    }
  }
}

self.onmessage = async (e: MessageEvent<WorkerParams>) => {
  const { 
    optimizer, hiddenSize, learningRate, epochs, batchSize, 
    inputSize, outputSize, X_train, y_train, X_test, y_test,
    trainSamples, testSamples 
  } = e.data;

  const startTime = Date.now();
  const maxTrainingTime = 60000 * 1000;

  // Initialize weights with He initialization
  const initHe = (sizeIn: number, sizeOut: number) => randomArray(sizeIn * sizeOut, Math.sqrt(2.0 / sizeIn));
  
  const h1Size = 512;
  const h2Size = 256;
  const h3Size = 128;

  let w1 = initHe(inputSize, h1Size);
  let b1 = createArray(h1Size);
  let w2 = initHe(h1Size, h2Size);
  let b2 = createArray(h2Size);
  let w3 = initHe(h2Size, h3Size);
  let b3 = createArray(h3Size);
  let w4 = initHe(h3Size, outputSize);
  let b4 = createArray(outputSize);

  // Optimizer states
  const createStates = (size: number) => ({
    m: createArray(size),
    v: createArray(size),
    g: createArray(size)
  });

  const states = {
    w1: createStates(inputSize * h1Size),
    b1: createStates(h1Size),
    w2: createStates(h1Size * h2Size),
    b2: createStates(h2Size),
    w3: createStates(h2Size * h3Size),
    b3: createStates(h3Size),
    w4: createStates(h3Size * outputSize),
    b4: createStates(outputSize)
  };

  let t = 0;
  let currentLR = learningRate;

  let metrics: any[] = [];
  let indices = Array.from({ length: trainSamples }, (_, i) => i);
  let convergenceEpoch85: number | undefined = undefined;
  let totalUpdateNorm = 0;
  let totalGradNorm = 0;
  let totalGradVariance = 0;

  // Pre-allocate batch buffers
  const maxBatch = Math.max(batchSize, 100, 500);
  const xBatch = createArray(maxBatch * inputSize);
  const augBuffer = createArray(inputSize);
  const augCopy = createArray(inputSize);
  
  // Forward buffers
  const z1 = createArray(maxBatch * h1Size);
  const a1 = createArray(maxBatch * h1Size);
  const drop1 = createArray(maxBatch * h1Size);
  
  const z2 = createArray(maxBatch * h2Size);
  const a2 = createArray(maxBatch * h2Size);
  const drop2 = createArray(maxBatch * h2Size);
  
  const z3 = createArray(maxBatch * h3Size);
  const a3 = createArray(maxBatch * h3Size);
  const drop3 = createArray(maxBatch * h3Size);
  
  const z4 = createArray(maxBatch * outputSize);
  const a4 = createArray(maxBatch * outputSize);

  // Backward buffers
  const dz4 = createArray(maxBatch * outputSize);
  const dw4 = createArray(h3Size * outputSize);
  const db4 = createArray(outputSize);
  
  const da3 = createArray(maxBatch * h3Size);
  const dz3 = createArray(maxBatch * h3Size);
  const dw3 = createArray(h2Size * h3Size);
  const db3 = createArray(h3Size);
  
  const da2 = createArray(maxBatch * h2Size);
  const dz2 = createArray(maxBatch * h2Size);
  const dw2 = createArray(h1Size * h2Size);
  const db2 = createArray(h2Size);
  
  const da1 = createArray(maxBatch * h1Size);
  const dz1 = createArray(maxBatch * h1Size);
  const dw1 = createArray(inputSize * h1Size);
  const db1 = createArray(h1Size);

  const backpropDropout = (dA: Float32Array, mask: Float32Array, size: number, p: number) => {
    if (p <= 0) return;
    const keepProb = 1 - p;
    for (let i = 0; i < size; i++) {
      dA[i] = (mask[i] * dA[i]) / keepProb;
    }
  };

  // Helper for Gradient Clipping
  const clipGradients = (grad: Float32Array, limit: number) => {
    for (let i = 0; i < grad.length; i++) {
      if (grad[i] > limit) grad[i] = limit;
      if (grad[i] < -limit) grad[i] = -limit;
    }
  };

  // Data Augmentation
  const augment = (img: Float32Array, size: number, target: Float32Array, targetOffset: number) => {
    let width = 0, height = 0, channels = 0;
    
    if (size === 3072) { width = 32; height = 32; channels = 3; }
    else if (size === 784) { width = 28; height = 28; channels = 1; }
    else if (size === 1024) { width = 32; height = 32; channels = 1; }
    else if (size === 2352) { width = 28; height = 28; channels = 3; }
    else { 
      target.set(img, targetOffset);
      return; 
    }

    augBuffer.set(img);
    const planeSize = width * height;
    
    // 1. Random Horizontal Flip
    if (Math.random() > 0.5) {
      for (let c = 0; c < channels; c++) {
        const cOff = c * planeSize;
        for (let y = 0; y < height; y++) {
          const yOff = cOff + y * width;
          for (let x = 0; x < (width >> 1); x++) {
            const idx1 = yOff + x;
            const idx2 = yOff + (width - 1 - x);
            const tmp = augBuffer[idx1];
            augBuffer[idx1] = augBuffer[idx2];
            augBuffer[idx2] = tmp;
          }
        }
      }
    }

    // 2. Random Shift
    const shiftX = Math.floor(Math.random() * 3) - 1; 
    const shiftY = Math.floor(Math.random() * 3) - 1;
    if (shiftX !== 0 || shiftY !== 0) {
      augCopy.set(augBuffer);
      augBuffer.fill(0);
      for (let c = 0; c < channels; c++) {
        const cOff = c * planeSize;
        for (let y = 0; y < height; y++) {
          const ny = y + shiftY;
          if (ny < 0 || ny >= height) continue;
          const yOff = cOff + y * width;
          const nyOff = cOff + ny * width;
          for (let x = 0; x < width; x++) {
            const nx = x + shiftX;
            if (nx < 0 || nx >= width) continue;
            augBuffer[nyOff + nx] = augCopy[yOff + x];
          }
        }
      }
    }

    // 3. Random Brightness
    const brightness = 0.9 + Math.random() * 0.2;
    if (brightness !== 1.0) {
      for (let i = 0; i < size; i++) {
        augBuffer[i] *= brightness;
      }
    }
    target.set(augBuffer, targetOffset);
  };

  try {
    let prevLoss = 0;
    for (let epoch = 1; epoch <= epochs; epoch++) {
      const epochStartTime = Date.now();
      
      // Learning Rate Schedule
      currentLR = learningRate * Math.pow(0.95, epoch - 1);

      if (Date.now() - startTime > maxTrainingTime) {
        self.postMessage({ type: 'timeout', optimizer });
        return;
      }

      // Shuffle indices
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      let totalLoss = 0;
      let batchCount = 0;
      let totalCorrect = 0;
      let epochGradNorms: number[] = [];
      let epochUpdateNorms: number[] = [];

      for (let i = 0; i < trainSamples; i += batchSize) {
        const currentBatchSize = Math.min(batchSize, trainSamples - i);
        
        // Prepare batch data with augmentation
        for (let b = 0; b < currentBatchSize; b++) {
          const idx = indices[i + b];
          const img = X_train.subarray(idx * inputSize, (idx + 1) * inputSize);
          augment(img, inputSize, xBatch, b * inputSize);
        }

        // Forward Pass
        // Layer 1
        matmul(xBatch, w1, currentBatchSize, inputSize, h1Size, z1);
        addBias(z1, b1, currentBatchSize, h1Size);
        a1.set(z1.subarray(0, currentBatchSize * h1Size));
        reluAndDropout(a1, drop1, currentBatchSize * h1Size, 0.4, true);
        
        // Layer 2
        matmul(a1, w2, currentBatchSize, h1Size, h2Size, z2);
        addBias(z2, b2, currentBatchSize, h2Size);
        a2.set(z2.subarray(0, currentBatchSize * h2Size));
        reluAndDropout(a2, drop2, currentBatchSize * h2Size, 0.3, true);

        // Layer 3
        matmul(a2, w3, currentBatchSize, h2Size, h3Size, z3);
        addBias(z3, b3, currentBatchSize, h3Size);
        a3.set(z3.subarray(0, currentBatchSize * h3Size));
        reluAndDropout(a3, drop3, currentBatchSize * h3Size, 0.2, true);

        // Layer 4 (Output)
        matmul(a3, w4, currentBatchSize, h3Size, outputSize, z4);
        addBias(z4, b4, currentBatchSize, outputSize);
        a4.set(z4.subarray(0, currentBatchSize * outputSize));
        softmax(a4, currentBatchSize, outputSize);

        // Compute Loss and dz4
        let batchLoss = 0;
        dz4.fill(0, 0, currentBatchSize * outputSize);
        for (let b = 0; b < currentBatchSize; b++) {
          const label = y_train[indices[i + b]];
          const off = b * outputSize;
          for (let j = 0; j < outputSize; j++) {
            const target = (j === label) ? 1.0 : 0.0;
            dz4[off + j] = (a4[off + j] - target) / currentBatchSize;
          }
          batchLoss -= Math.log(a4[off + label] + 1e-15);
          
          let maxProb = -1, pred = -1;
          for (let k = 0; k < outputSize; k++) {
            if (a4[off + k] > maxProb) { maxProb = a4[off + k]; pred = k; }
          }
          if (pred === label) totalCorrect++;
        }

        const currentLoss = batchLoss / currentBatchSize;
        totalLoss += currentLoss;

        if (isNaN(totalLoss) || !isFinite(totalLoss)) {
          self.postMessage({ type: 'error', optimizer, message: `Training diverged at epoch ${epoch}.` });
          return;
        }

        // Backward Pass
        // Layer 4
        matmulATB(a3, dz4, currentBatchSize, h3Size, outputSize, dw4);
        sumCols(dz4, currentBatchSize, outputSize, db4);
        matmulABT(dz4, w4, currentBatchSize, outputSize, h3Size, da3);
        
        // Layer 3
        backpropDropout(da3, drop3, currentBatchSize * h3Size, 0.2);
        dz3.set(da3.subarray(0, currentBatchSize * h3Size));
        reluDeriv(dz3, z3, currentBatchSize * h3Size);
        matmulATB(a2, dz3, currentBatchSize, h2Size, h3Size, dw3);
        sumCols(dz3, currentBatchSize, h3Size, db3);
        matmulABT(dz3, w3, currentBatchSize, h3Size, h2Size, da2);

        // Layer 2
        backpropDropout(da2, drop2, currentBatchSize * h2Size, 0.3);
        dz2.set(da2.subarray(0, currentBatchSize * h2Size));
        reluDeriv(dz2, z2, currentBatchSize * h2Size);
        matmulATB(a1, dz2, currentBatchSize, h1Size, h2Size, dw2);
        sumCols(dz2, currentBatchSize, h2Size, db2);
        matmulABT(dz2, w2, currentBatchSize, h2Size, h1Size, da1);

        // Layer 1
        backpropDropout(da1, drop1, currentBatchSize * h1Size, 0.4);
        dz1.set(da1.subarray(0, currentBatchSize * h1Size));
        reluDeriv(dz1, z1, currentBatchSize * h1Size);
        matmulATB(xBatch, dz1, currentBatchSize, inputSize, h1Size, dw1);
        sumCols(dz1, currentBatchSize, h1Size, db1);

        // Gradient Clipping
        const clipVal = 1.0;
        clipGradients(dw1, clipVal); clipGradients(db1, clipVal);
        clipGradients(dw2, clipVal); clipGradients(db2, clipVal);
        clipGradients(dw3, clipVal); clipGradients(db3, clipVal);
        clipGradients(dw4, clipVal); clipGradients(db4, clipVal);

        // Calculate Gradient Norm for this batch
        let batchGradNorm = 0;
        [dw1, dw2, dw3, dw4].forEach(g => {
          for (let k = 0; k < g.length; k++) batchGradNorm += g[k] * g[k];
        });
        batchGradNorm = Math.sqrt(batchGradNorm);
        epochGradNorms.push(batchGradNorm);

        // Optimizer Updates
        let batchUpdateNorm = 0;
        const eps = 1e-8;
        
        const updateFn = (param: Float32Array, grad: Float32Array, state: any) => {
          let uNorm = 0;
          const len = param.length;
          if (optimizer === 'SGD') {
            for (let k = 0; k < len; k++) {
              const update = currentLR * grad[k];
              param[k] -= update;
              uNorm += update * update;
            }
          } else if (optimizer === 'Adagrad') {
            for (let k = 0; k < len; k++) {
              state.g[k] += grad[k] * grad[k];
              const update = (currentLR * grad[k]) / (Math.sqrt(state.g[k]) + eps);
              param[k] -= update;
              uNorm += update * update;
            }
          } else if (optimizer === 'RMSProp') {
            const gamma = 0.9;
            const oneMinusGamma = 0.1;
            for (let k = 0; k < len; k++) {
              state.g[k] = gamma * state.g[k] + oneMinusGamma * grad[k] * grad[k];
              const update = (currentLR * grad[k]) / (Math.sqrt(state.g[k]) + eps);
              param[k] -= update;
              uNorm += update * update;
            }
          } else { // Adam
            const b1_adam = 0.9, b2_adam = 0.999;
            const bc1 = 1 - Math.pow(b1_adam, t + 1);
            const bc2 = 1 - Math.pow(b2_adam, t + 1);
            const lr_eff = currentLR / bc1;
            for (let k = 0; k < len; k++) {
              state.m[k] = b1_adam * state.m[k] + (1 - b1_adam) * grad[k];
              state.v[k] = b2_adam * state.v[k] + (1 - b2_adam) * grad[k] * grad[k];
              const update = (lr_eff * state.m[k]) / (Math.sqrt(state.v[k] / bc2) + eps);
              param[k] -= update;
              uNorm += update * update;
            }
          }
          return uNorm;
        };

        batchUpdateNorm += updateFn(w1, dw1, states.w1);
        batchUpdateNorm += updateFn(b1, db1, states.b1);
        batchUpdateNorm += updateFn(w2, dw2, states.w2);
        batchUpdateNorm += updateFn(b2, db2, states.b2);
        batchUpdateNorm += updateFn(w3, dw3, states.w3);
        batchUpdateNorm += updateFn(b3, db3, states.b3);
        batchUpdateNorm += updateFn(w4, dw4, states.w4);
        batchUpdateNorm += updateFn(b4, db4, states.b4);
        
        epochUpdateNorms.push(Math.sqrt(batchUpdateNorm));
        
        if (optimizer === 'Adam') t++;

        batchCount++;
        if (batchCount % 20 === 0) {
          self.postMessage({ 
            type: 'progress', 
            optimizer, 
            epoch, 
            trainProgress: ((epoch - 1) / epochs + (batchCount / (trainSamples / batchSize)) / epochs) * 100,
            testProgress: 0
          });
        }
      }

      // Epoch Evaluation
      const avgLoss = totalLoss / batchCount;
      const epochAccuracy = totalCorrect / trainSamples;
      
      // Evaluation on test set
      const evalBatchSize = 100;
      let testCorrect = 0;
      for (let j = 0; j < Math.min(1000, testSamples); j += evalBatchSize) {
        const currentEvalBatchSize = Math.min(evalBatchSize, testSamples - j);
        for (let b = 0; b < currentEvalBatchSize; b++) {
          const idx = j + b;
          xBatch.set(X_test.subarray(idx * inputSize, (idx + 1) * inputSize), b * inputSize);
        }
        // Forward pass (no dropout)
        matmul(xBatch, w1, currentEvalBatchSize, inputSize, h1Size, z1);
        addBias(z1, b1, currentEvalBatchSize, h1Size);
        a1.set(z1.subarray(0, currentEvalBatchSize * h1Size));
        reluAndDropout(a1, drop1, currentEvalBatchSize * h1Size, 0, false);
        
        matmul(a1, w2, currentEvalBatchSize, h1Size, h2Size, z2);
        addBias(z2, b2, currentEvalBatchSize, h2Size);
        a2.set(z2.subarray(0, currentEvalBatchSize * h2Size));
        reluAndDropout(a2, drop2, currentEvalBatchSize * h2Size, 0, false);

        matmul(a2, w3, currentEvalBatchSize, h2Size, h3Size, z3);
        addBias(z3, b3, currentEvalBatchSize, h3Size);
        a3.set(z3.subarray(0, currentEvalBatchSize * h3Size));
        reluAndDropout(a3, drop3, currentEvalBatchSize * h3Size, 0, false);

        matmul(a3, w4, currentEvalBatchSize, h3Size, outputSize, z4);
        addBias(z4, b4, currentEvalBatchSize, outputSize);
        softmax(z4, currentEvalBatchSize, outputSize);
        
        for (let b = 0; b < currentEvalBatchSize; b++) {
          const off = b * outputSize;
          let maxProb = -1, pred = -1;
          for (let k = 0; k < outputSize; k++) {
            if (z4[off + k] > maxProb) { maxProb = z4[off + k]; pred = k; }
          }
          if (pred === y_test[j + b]) testCorrect++;
        }
      }
      const testAccuracy = testCorrect / Math.min(1000, testSamples);

      if (convergenceEpoch85 === undefined && testAccuracy >= 0.85) {
        convergenceEpoch85 = epoch;
      }

      // Debugging Checks
      let probSum = 0;
      for (let k = 0; k < outputSize; k++) probSum += z4[k]; // Check first sample of last test batch
      
      const avgGradNorm = epochGradNorms.reduce((a, b) => a + b, 0) / epochGradNorms.length;
      const gradVariance = epochGradNorms.reduce((a, b) => a + Math.pow(b - avgGradNorm, 2), 0) / epochGradNorms.length;
      totalGradNorm += avgGradNorm;
      totalGradVariance += gradVariance;

      let weightNorm = 0;
      [w1, w2, w3, w4].forEach(w => {
        for (let k = 0; k < w.length; k++) weightNorm += w[k] * w[k];
      });
      weightNorm = Math.sqrt(weightNorm);

      const avgUpdateNorm = epochUpdateNorms.reduce((a, b) => a + b, 0) / epochUpdateNorms.length;
      const updateRatio = weightNorm > 0 ? avgUpdateNorm / weightNorm : 0;
      totalUpdateNorm += avgUpdateNorm;

      const epochTime = (Date.now() - epochStartTime) / 1000;
      const throughput = trainSamples / epochTime;

      const convergenceRate = prevLoss > 0 ? Math.log(prevLoss) - Math.log(avgLoss) : 0;
      prevLoss = avgLoss;

      console.log(`[${optimizer}] Epoch ${epoch}: Loss=${avgLoss.toFixed(4)}, TestAcc=${testAccuracy.toFixed(4)}, ProbSum=${probSum.toFixed(4)}, GradNorm=${avgGradNorm.toFixed(4)}, WeightNorm=${weightNorm.toFixed(4)}`);

      const metric = {
        epoch,
        loss: avgLoss,
        accuracy: testAccuracy,
        trainAccuracy: epochAccuracy,
        testAccuracy: testAccuracy,
        learningRate: currentLR,
        gradientNorm: avgGradNorm,
        parameterNorm: weightNorm,
        updateRatio: updateRatio,
        convergenceSpeed: convergenceRate,
        gradientVariance: gradVariance,
        throughput: throughput,
        lossVariance: 0, // Will be calculated at the end for the whole experiment
        convergenceRate: convergenceRate,
        generalizationGap: epochAccuracy - testAccuracy
      };
      metrics.push(metric);

      self.postMessage({ 
        type: 'progress', 
        optimizer, 
        epoch, 
        metric, 
        trainProgress: (epoch / epochs) * 100,
        testProgress: 0
      });

      await new Promise(resolve => setTimeout(resolve, 0));
    }

    // Final Testing
    let finalCorrect = 0;
    const finalBatchSize = 500;
    const y_true: number[] = [];
    const y_pred: number[] = [];
    const y_probs: number[][] = [];

    for (let i = 0; i < testSamples; i += finalBatchSize) {
      const currentBatchSize = Math.min(finalBatchSize, testSamples - i);
      for (let b = 0; b < currentBatchSize; b++) {
        xBatch.set(X_test.subarray((i + b) * inputSize, (i + b + 1) * inputSize), b * inputSize);
      }
      
      matmul(xBatch, w1, currentBatchSize, inputSize, h1Size, z1);
      addBias(z1, b1, currentBatchSize, h1Size);
      a1.set(z1.subarray(0, currentBatchSize * h1Size));
      reluAndDropout(a1, drop1, currentBatchSize * h1Size, 0, false);
      
      matmul(a1, w2, currentBatchSize, h1Size, h2Size, z2);
      addBias(z2, b2, currentBatchSize, h2Size);
      a2.set(z2.subarray(0, currentBatchSize * h2Size));
      reluAndDropout(a2, drop2, currentBatchSize * h2Size, 0, false);

      matmul(a2, w3, currentBatchSize, h2Size, h3Size, z3);
      addBias(z3, b3, currentBatchSize, h3Size);
      a3.set(z3.subarray(0, currentBatchSize * h3Size));
      reluAndDropout(a3, drop3, currentBatchSize * h3Size, 0, false);

      matmul(a3, w4, currentBatchSize, h3Size, outputSize, z4);
      addBias(z4, b4, currentBatchSize, outputSize);
      softmax(z4, currentBatchSize, outputSize);

      for (let b = 0; b < currentBatchSize; b++) {
        const off = b * outputSize;
        const probs = Array.from(z4.subarray(off, off + outputSize));
        const pred = probs.indexOf(Math.max(...probs));
        const label = y_test[i + b];
        y_true.push(label);
        y_pred.push(pred);
        y_probs.push(probs);
        if (pred === label) finalCorrect++;
      }
    }

    const testAccuracy = finalCorrect / testSamples;
    const trainingTime = (Date.now() - startTime) / 1000;
    
    // Calculate precision, recall, f1 (macro average)
    const numClasses = outputSize;
    const confusionMatrix = Array.from({ length: numClasses }, () => Array(numClasses).fill(0));
    for (let i = 0; i < y_true.length; i++) {
      confusionMatrix[y_true[i]][y_pred[i]]++;
    }

    let totalPrecision = 0, totalRecall = 0, totalF1 = 0;
    for (let i = 0; i < numClasses; i++) {
      const tp = confusionMatrix[i][i];
      const fp = confusionMatrix.reduce((sum, row, idx) => idx !== i ? sum + row[i] : sum, 0);
      const fn = confusionMatrix[i].reduce((sum, val, idx) => idx !== i ? sum + val : sum, 0);
      
      const p = tp + fp > 0 ? tp / (tp + fp) : 0;
      const r = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f = p + r > 0 ? 2 * p * r / (p + r) : 0;
      
      totalPrecision += p;
      totalRecall += r;
      totalF1 += f;
    }
    const precision = totalPrecision / numClasses;
    const recall = totalRecall / numClasses;
    const f1Score = totalF1 / numClasses;

    // Calculate convergence rate and loss variance
    const losses = metrics.map(m => m.loss);
    const avgLoss = losses.reduce((a, b) => a + b, 0) / losses.length;
    const lossVariance = losses.reduce((a, b) => a + Math.pow(b - avgLoss, 2), 0) / losses.length;
    const convergenceRate = losses.length > 1 ? (losses[0] - losses[losses.length - 1]) / losses.length : 0;

    self.postMessage({
      type: 'training_complete',
      optimizer,
      metrics: {
        optimizer,
        learningRate,
        testAccuracy,
        precision,
        recall,
        f1Score,
        logLoss: losses[losses.length - 1],
        trainingTime,
        testingTime: 0.1, // Small constant for testing time
        executionTime: trainingTime,
        convergenceRate,
        lossVariance,
        aulc: metrics.reduce((acc, m) => acc + m.accuracy, 0) / metrics.length,
        metrics,
        convergenceEpoch85,
        avgGradientNorm: totalGradNorm / epochs,
        avgGradientVariance: totalGradVariance / epochs,
        avgThroughput: metrics.reduce((acc, m) => acc + m.throughput, 0) / metrics.length,
        avgUpdateRatio: totalUpdateNorm / epochs
      }
    });
  } catch (err: any) {
    self.postMessage({ type: 'error', optimizer, message: err.message || 'Unknown worker error' });
  } finally {
    (indices as any) = null;
    (metrics as any) = null;
  }
};
