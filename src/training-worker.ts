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

// Matrix Transpose Multiplication: C = A^T * B
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

// Matrix Transpose Multiplication: C = A * B^T
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

function addBias(A: Float32Array, b: Float32Array, rows: number, cols: number) {
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    for (let j = 0; j < cols; j++) {
      A[off + j] += b[j];
    }
  }
}

function relu(A: Float32Array, size: number) {
  for (let i = 0; i < size; i++) {
    if (A[i] < 0) A[i] = 0;
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

  // Initialize weights
  const scale1 = Math.sqrt(2.0 / inputSize);
  const scale2 = Math.sqrt(2.0 / hiddenSize);
  
  let w1 = randomArray(inputSize * hiddenSize, scale1);
  let b1 = createArray(hiddenSize);
  let w2 = randomArray(hiddenSize * outputSize, scale2);
  let b2 = createArray(outputSize);

  // Optimizer states
  let g_w1 = createArray(inputSize * hiddenSize);
  let g_b1 = createArray(hiddenSize);
  let g_w2 = createArray(hiddenSize * outputSize);
  let g_b2 = createArray(outputSize);

  let m_w1 = createArray(inputSize * hiddenSize);
  let m_b1 = createArray(hiddenSize);
  let m_w2 = createArray(hiddenSize * outputSize);
  let m_b2 = createArray(outputSize);

  let v_w1 = createArray(inputSize * hiddenSize);
  let v_b1 = createArray(hiddenSize);
  let v_w2 = createArray(hiddenSize * outputSize);
  let v_b2 = createArray(outputSize);

  let t = 0;

  let metrics: any[] = [];
  let indices = Array.from({ length: trainSamples }, (_, i) => i);

  // Pre-allocate batch buffers (Step 7)
  // Ensure buffers are large enough for training batch size, evaluation batch size (100), and test batch size (500)
  const maxBatch = Math.max(batchSize, 100, 500);
  const xBatch = createArray(maxBatch * inputSize);
  const z1 = createArray(maxBatch * hiddenSize);
  const a1 = createArray(maxBatch * hiddenSize);
  const z2 = createArray(maxBatch * outputSize);
  const a2 = createArray(maxBatch * outputSize);
  const dz2 = createArray(maxBatch * outputSize);
  const dw2 = createArray(hiddenSize * outputSize);
  const db2 = createArray(outputSize);
  const da1 = createArray(maxBatch * hiddenSize);
  const dz1 = createArray(maxBatch * hiddenSize);
  const dw1 = createArray(inputSize * hiddenSize);
  const db1 = createArray(hiddenSize);

  // Pre-allocate update buffers
  const u_w1 = createArray(inputSize * hiddenSize);
  const u_b1 = createArray(hiddenSize);
  const u_w2 = createArray(hiddenSize * outputSize);
  const u_b2 = createArray(outputSize);

  try {
    for (let epoch = 1; epoch <= epochs; epoch++) {
      const epochStartTime = Date.now();
      
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
      let totalGradNorm = 0;
      let totalUpdateNorm = 0;
      const batchLosses: number[] = [];
      const batchGradNorms: number[] = [];

      for (let i = 0; i < trainSamples; i += batchSize) {
        const currentBatchSize = Math.min(batchSize, trainSamples - i);
        
        // Prepare batch data
        for (let b = 0; b < currentBatchSize; b++) {
          const idx = indices[i + b];
          xBatch.set(X_train.subarray(idx * inputSize, (idx + 1) * inputSize), b * inputSize);
        }

        // Forward Pass
        matmul(xBatch, w1, currentBatchSize, inputSize, hiddenSize, z1);
        addBias(z1, b1, currentBatchSize, hiddenSize);
        a1.set(z1.subarray(0, currentBatchSize * hiddenSize));
        relu(a1, currentBatchSize * hiddenSize);
        
        matmul(a1, w2, currentBatchSize, hiddenSize, outputSize, z2);
        addBias(z2, b2, currentBatchSize, outputSize);
        a2.set(z2.subarray(0, currentBatchSize * outputSize));
        softmax(a2, currentBatchSize, outputSize);

        // Compute Loss and dz2
        let batchLoss = 0;
        const smoothVal = 0.01 / outputSize;
        const targetVal = 0.99 + smoothVal;
        
        dz2.fill(0, 0, currentBatchSize * outputSize);
        for (let b = 0; b < currentBatchSize; b++) {
          const label = y_train[indices[i + b]];
          const off = b * outputSize;
          for (let j = 0; j < outputSize; j++) {
            const target = (j === label) ? targetVal : smoothVal;
            dz2[off + j] = (a2[off + j] - target) / currentBatchSize;
          }
          batchLoss -= Math.log(a2[off + label] + 1e-15);
        }

        const currentLoss = batchLoss / currentBatchSize;
        totalLoss += currentLoss;
        batchLosses.push(currentLoss);

        if (isNaN(totalLoss) || !isFinite(totalLoss)) {
          self.postMessage({ type: 'error', optimizer, message: `Training diverged at epoch ${epoch}.` });
          return;
        }

        // Backward Pass
        matmulATB(a1, dz2, currentBatchSize, hiddenSize, outputSize, dw2);
        sumCols(dz2, currentBatchSize, outputSize, db2);

        matmulABT(dz2, w2, currentBatchSize, outputSize, hiddenSize, da1);
        dz1.set(da1.subarray(0, currentBatchSize * hiddenSize));
        reluDeriv(dz1, z1, currentBatchSize * hiddenSize);
        matmulATB(xBatch, dz1, currentBatchSize, inputSize, hiddenSize, dw1);
        sumCols(dz1, currentBatchSize, hiddenSize, db1);

        // Optimizer Updates
        const eps = 1e-8;

        if (optimizer === 'SGD') {
          for (let k = 0; k < dw1.length; k++) u_w1[k] = learningRate * dw1[k];
          for (let k = 0; k < db1.length; k++) u_b1[k] = learningRate * db1[k];
          for (let k = 0; k < dw2.length; k++) u_w2[k] = learningRate * dw2[k];
          for (let k = 0; k < db2.length; k++) u_b2[k] = learningRate * db2[k];
        } else if (optimizer === 'Adagrad') {
          for (let k = 0; k < dw1.length; k++) { g_w1[k] += dw1[k] * dw1[k]; u_w1[k] = (learningRate * dw1[k]) / Math.sqrt(g_w1[k] + eps); }
          for (let k = 0; k < db1.length; k++) { g_b1[k] += db1[k] * db1[k]; u_b1[k] = (learningRate * db1[k]) / Math.sqrt(g_b1[k] + eps); }
          for (let k = 0; k < dw2.length; k++) { g_w2[k] += dw2[k] * dw2[k]; u_w2[k] = (learningRate * dw2[k]) / Math.sqrt(g_w2[k] + eps); }
          for (let k = 0; k < db2.length; k++) { g_b2[k] += db2[k] * db2[k]; u_b2[k] = (learningRate * db2[k]) / Math.sqrt(g_b2[k] + eps); }
        } else if (optimizer === 'RMSProp') {
          const gamma = 0.9;
          for (let k = 0; k < dw1.length; k++) { g_w1[k] = gamma * g_w1[k] + (1 - gamma) * dw1[k] * dw1[k]; u_w1[k] = (learningRate * dw1[k]) / Math.sqrt(g_w1[k] + eps); }
          for (let k = 0; k < db1.length; k++) { g_b1[k] = gamma * g_b1[k] + (1 - gamma) * db1[k] * db1[k]; u_b1[k] = (learningRate * db1[k]) / Math.sqrt(g_b1[k] + eps); }
          for (let k = 0; k < dw2.length; k++) { g_w2[k] = gamma * g_w2[k] + (1 - gamma) * dw2[k] * dw2[k]; u_w2[k] = (learningRate * dw2[k]) / Math.sqrt(g_w2[k] + eps); }
          for (let k = 0; k < db2.length; k++) { g_b2[k] = gamma * g_b2[k] + (1 - gamma) * db2[k] * db2[k]; u_b2[k] = (learningRate * db2[k]) / Math.sqrt(g_b2[k] + eps); }
        } else { // Adam
          const b1_adam = 0.9, b2_adam = 0.999;
          t++;
          const bc1 = 1 - Math.pow(b1_adam, t);
          const bc2 = 1 - Math.pow(b2_adam, t);
          for (let k = 0; k < dw1.length; k++) {
            m_w1[k] = b1_adam * m_w1[k] + (1 - b1_adam) * dw1[k];
            v_w1[k] = b2_adam * v_w1[k] + (1 - b2_adam) * dw1[k] * dw1[k];
            u_w1[k] = (learningRate * (m_w1[k] / bc1)) / (Math.sqrt(v_w1[k] / bc2) + eps);
          }
          for (let k = 0; k < db1.length; k++) {
            m_b1[k] = b1_adam * m_b1[k] + (1 - b1_adam) * db1[k];
            v_b1[k] = b2_adam * v_b1[k] + (1 - b2_adam) * db1[k] * db1[k];
            u_b1[k] = (learningRate * (m_b1[k] / bc1)) / (Math.sqrt(v_b1[k] / bc2) + eps);
          }
          for (let k = 0; k < dw2.length; k++) {
            m_w2[k] = b1_adam * m_w2[k] + (1 - b1_adam) * dw2[k];
            v_w2[k] = b2_adam * v_w2[k] + (1 - b2_adam) * dw2[k] * dw2[k];
            u_w2[k] = (learningRate * (m_w2[k] / bc1)) / (Math.sqrt(v_w2[k] / bc2) + eps);
          }
          for (let k = 0; k < db2.length; k++) {
            m_b2[k] = b1_adam * m_b2[k] + (1 - b1_adam) * db2[k];
            v_b2[k] = b2_adam * v_b2[k] + (1 - b2_adam) * db2[k] * db2[k];
            u_b2[k] = (learningRate * (m_b2[k] / bc1)) / (Math.sqrt(v_b2[k] / bc2) + eps);
          }
        }

        // Track Gradient Norm
        let gradSqSum = 0;
        for (let k = 0; k < dw1.length; k++) gradSqSum += dw1[k] * dw1[k];
        for (let k = 0; k < dw2.length; k++) gradSqSum += dw2[k] * dw2[k];
        for (let k = 0; k < db1.length; k++) gradSqSum += db1[k] * db1[k];
        for (let k = 0; k < db2.length; k++) gradSqSum += db2[k] * db2[k];
        const gradNorm = Math.sqrt(gradSqSum);
        totalGradNorm += gradNorm;
        batchGradNorms.push(gradNorm);

        // Track Update Ratio
        let updateSqSum = 0;
        let weightSqSum = 0;
        for (let k = 0; k < u_w1.length; k++) { updateSqSum += u_w1[k] * u_w1[k]; weightSqSum += w1[k] * w1[k]; }
        for (let k = 0; k < u_w2.length; k++) { updateSqSum += u_w2[k] * u_w2[k]; weightSqSum += w2[k] * w2[k]; }
        totalUpdateNorm += Math.sqrt(updateSqSum) / (Math.sqrt(weightSqSum) + 1e-12);

        // Update Weights
        for (let k = 0; k < w1.length; k++) w1[k] -= u_w1[k];
        for (let k = 0; k < b1.length; k++) b1[k] -= u_b1[k];
        for (let k = 0; k < w2.length; k++) w2[k] -= u_w2[k];
        for (let k = 0; k < b2.length; k++) b2[k] -= u_b2[k];

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
      let epochDuration = Date.now() - epochStartTime;
      const throughput = trainSamples / (epochDuration / 1000);

      const lossVariance = batchLosses.reduce((sum, l) => sum + Math.pow(l - avgLoss, 2), 0) / batchCount;
      const avgGradNorm = totalGradNorm / batchCount;
      const gradientVariance = batchGradNorms.reduce((sum, g) => sum + Math.pow(g - avgGradNorm, 2), 0) / batchCount;

      let paramSqSum = 0;
      for (let k = 0; k < w1.length; k++) paramSqSum += w1[k] * w1[k];
      for (let k = 0; k < w2.length; k++) paramSqSum += w2[k] * w2[k];
      for (let k = 0; k < b1.length; k++) paramSqSum += b1[k] * b1[k];
      for (let k = 0; k < b2.length; k++) paramSqSum += b2[k] * b2[k];
      const parameterNorm = Math.sqrt(paramSqSum);

      // Fast eval for training accuracy
      const evalSize = Math.min(1000, trainSamples);
      let trainCorrect = 0;
      const evalBatchSize = 100;
      for (let j = 0; j < evalSize; j += evalBatchSize) {
        const currentEvalBatchSize = Math.min(evalBatchSize, evalSize - j);
        for (let b = 0; b < currentEvalBatchSize; b++) {
          const idx = Math.floor(Math.random() * trainSamples);
          xBatch.set(X_train.subarray(idx * inputSize, (idx + 1) * inputSize), b * inputSize);
          dz2[b] = y_train[idx]; // Reuse dz2[b] as temporary storage for labels
        }
        matmul(xBatch, w1, currentEvalBatchSize, inputSize, hiddenSize, z1);
        addBias(z1, b1, currentEvalBatchSize, hiddenSize);
        a1.set(z1.subarray(0, currentEvalBatchSize * hiddenSize));
        relu(a1, currentEvalBatchSize * hiddenSize);
        matmul(a1, w2, currentEvalBatchSize, hiddenSize, outputSize, z2);
        addBias(z2, b2, currentEvalBatchSize, outputSize);
        softmax(z2, currentEvalBatchSize, outputSize);
        
        for (let b = 0; b < currentEvalBatchSize; b++) {
          const off = b * outputSize;
          let maxProb = -1;
          let predLabel = -1;
          for (let k = 0; k < outputSize; k++) {
            if (z2[off + k] > maxProb) {
              maxProb = z2[off + k];
              predLabel = k;
            }
          }
          if (predLabel === dz2[b]) trainCorrect++;
        }
      }
      const trainAccuracy = trainCorrect / evalSize;

      // Fast eval for testing accuracy
      const testEvalSize = Math.min(1000, testSamples);
      let testCorrect = 0;
      for (let j = 0; j < testEvalSize; j += evalBatchSize) {
        const currentEvalBatchSize = Math.min(evalBatchSize, testEvalSize - j);
        for (let b = 0; b < currentEvalBatchSize; b++) {
          const idx = Math.floor(Math.random() * testSamples);
          xBatch.set(X_test.subarray(idx * inputSize, (idx + 1) * inputSize), b * inputSize);
          dz2[b] = y_test[idx];
        }
        matmul(xBatch, w1, currentEvalBatchSize, inputSize, hiddenSize, z1);
        addBias(z1, b1, currentEvalBatchSize, hiddenSize);
        a1.set(z1.subarray(0, currentEvalBatchSize * hiddenSize));
        relu(a1, currentEvalBatchSize * hiddenSize);
        matmul(a1, w2, currentEvalBatchSize, hiddenSize, outputSize, z2);
        addBias(z2, b2, currentEvalBatchSize, outputSize);
        softmax(z2, currentEvalBatchSize, outputSize);
        
        for (let b = 0; b < currentEvalBatchSize; b++) {
          const off = b * outputSize;
          let maxProb = -1;
          let predLabel = -1;
          for (let k = 0; k < outputSize; k++) {
            if (z2[off + k] > maxProb) {
              maxProb = z2[off + k];
              predLabel = k;
            }
          }
          if (predLabel === dz2[b]) testCorrect++;
        }
      }
      const testAccuracy = testCorrect / testEvalSize;

      const metric = {
        epoch,
        loss: avgLoss,
        accuracy: testAccuracy,
        trainAccuracy,
        testAccuracy,
        gradientNorm: avgGradNorm,
        updateRatio: totalUpdateNorm / batchCount,
        convergenceSpeed: metrics.length > 0 ? metrics[metrics.length - 1].loss - avgLoss : 0,
        gradientVariance,
        parameterNorm,
        throughput,
        lossVariance
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
    const testStartTime = Date.now();
    const y_true: number[] = [];
    const y_pred: number[] = [];
    const y_probs: number[][] = [];
    
    const testBatchSize = 500;
    for (let i = 0; i < testSamples; i += testBatchSize) {
      const currentBatchSize = Math.min(testBatchSize, testSamples - i);
      for (let b = 0; b < currentBatchSize; b++) {
        const idx = i + b;
        xBatch.set(X_test.subarray(idx * inputSize, (idx + 1) * inputSize), b * inputSize);
      }

      matmul(xBatch, w1, currentBatchSize, inputSize, hiddenSize, z1);
      addBias(z1, b1, currentBatchSize, hiddenSize);
      a1.set(z1.subarray(0, currentBatchSize * hiddenSize));
      relu(a1, currentBatchSize * hiddenSize);
      matmul(a1, w2, currentBatchSize, hiddenSize, outputSize, z2);
      addBias(z2, b2, currentBatchSize, outputSize);
      softmax(z2, currentBatchSize, outputSize);

      for (let b = 0; b < currentBatchSize; b++) {
        const off = b * outputSize;
        const probs = Array.from(z2.subarray(off, off + outputSize));
        const predLabel = probs.indexOf(Math.max(...probs));
        const trueLabel = y_test[i + b];
        y_true.push(trueLabel);
        y_pred.push(predLabel);
        y_probs.push(probs);
      }

      self.postMessage({
        type: 'progress',
        optimizer,
        epoch: epochs,
        trainProgress: 100,
        testProgress: Math.min(100, ((i + currentBatchSize) / testSamples) * 100)
      });
    }

    let totalCorrect = 0;
    for (let i = 0; i < testSamples; i++) {
      if (y_true[i] === y_pred[i]) totalCorrect++;
    }
    const testAccuracy = totalCorrect / testSamples;

    const classStats = Array.from({ length: outputSize }, () => ({ tp: 0, fp: 0, fn: 0, support: 0 }));
    for (let i = 0; i < testSamples; i++) {
      const t = y_true[i];
      const p = y_pred[i];
      if (t < outputSize) classStats[t].support++;
      if (t === p && t < outputSize) {
        classStats[t].tp++;
      } else {
        if (p < outputSize) classStats[p].fp++;
        if (t < outputSize) classStats[t].fn++;
      }
    }

    let macroPrecision = 0, macroRecall = 0, macroF1 = 0, classesWithSupport = 0;
    for (let i = 0; i < outputSize; i++) {
      const { tp, fp, fn, support } = classStats[i];
      if (support > 0) {
        const p = (tp + fp) > 0 ? tp / (tp + fp) : 0;
        const r = support > 0 ? tp / support : 0;
        const f = (p + r) > 0 ? (2 * p * r) / (p + r) : 0;
        macroPrecision += p; macroRecall += r; macroF1 += f;
        classesWithSupport++;
      }
    }

    const precision = classesWithSupport > 0 ? macroPrecision / classesWithSupport : 0;
    const recall = classesWithSupport > 0 ? macroRecall / classesWithSupport : 0;
    const f1Score = classesWithSupport > 0 ? macroF1 / classesWithSupport : 0;

    let totalLogLoss = 0;
    for (let i = 0; i < testSamples; i++) {
      totalLogLoss -= Math.log(y_probs[i][y_true[i]] + 1e-15);
    }
    const logLoss = totalLogLoss / testSamples;

    const convergenceRate = metrics.length > 1 && metrics[0].loss > 0 ? (metrics[0].loss - metrics[metrics.length - 1].loss) / metrics[0].loss : 0;
    const avgLossFinal = metrics.reduce((sum, m) => sum + m.loss, 0) / metrics.length;
    const lossVarianceFinal = metrics.reduce((sum, m) => sum + Math.pow(m.loss - avgLossFinal, 2), 0) / metrics.length;
    const trainingTime = (testStartTime - startTime) / 1000;
    const testingTime = (Date.now() - testStartTime) / 1000;

    self.postMessage({
      type: 'training_complete',
      optimizer,
      metrics: {
        optimizer,
        learningRate,
        metrics,
        testAccuracy,
        precision,
        recall,
        f1Score,
        logLoss,
        trainingTime,
        testingTime,
        executionTime: trainingTime + testingTime,
        convergenceRate,
        aulc: metrics.length > 1 ? metrics.reduce((acc, m, i) => i === 0 ? acc : acc + (metrics[i-1].accuracy + m.accuracy) / 2, 0) / (metrics.length - 1) : (metrics.length === 1 ? metrics[0].accuracy : 0),
        lossVariance: lossVarianceFinal
      }
    });
  } catch (err: any) {
    self.postMessage({ type: 'error', optimizer, message: err.message || 'Unknown worker error' });
  } finally {
    (indices as any) = null;
    (metrics as any) = null;
  }
};
