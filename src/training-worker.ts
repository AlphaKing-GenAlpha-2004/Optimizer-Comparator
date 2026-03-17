import * as math from 'mathjs';

// Optimized Matrix Operations for Float32Array
// Note: mathjs is used for convenience, but we ensure it works with TypedArrays
// for maximum performance in the worker.

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

self.onmessage = async (e: MessageEvent<WorkerParams>) => {
  const { 
    optimizer, hiddenSize, learningRate, epochs, batchSize, 
    inputSize, outputSize, X_train, y_train, X_test, y_test,
    trainSamples, testSamples 
  } = e.data;

  const startTime = Date.now();
  const maxTrainingTime = 60000 * 1000; // 60,000 seconds safety limit

  // Initialize weights (Step 2)
  // He Initialization for ReLU (W1)
  const scale1 = Math.sqrt(2.0 / inputSize);
  // Xavier Initialization for Softmax (W2)
  const scale2 = Math.sqrt(1.0 / hiddenSize);
  
  let w1 = math.multiply(math.random([inputSize, hiddenSize], -1, 1), scale1);
  let b1 = math.zeros([1, hiddenSize]);
  let w2 = math.multiply(math.random([hiddenSize, outputSize], -1, 1), scale2);
  let b2 = math.zeros([1, outputSize]);

  // Optimizer states
  let g_w1 = math.zeros([inputSize, hiddenSize]);
  let g_b1 = math.zeros([1, hiddenSize]);
  let g_w2 = math.zeros([hiddenSize, outputSize]);
  let g_b2 = math.zeros([1, outputSize]);

  let m_w1 = math.zeros([inputSize, hiddenSize]);
  let m_b1 = math.zeros([1, hiddenSize]);
  let m_w2 = math.zeros([hiddenSize, outputSize]);
  let m_b2 = math.zeros([1, outputSize]);

  let v_w1 = math.zeros([inputSize, hiddenSize]);
  let v_b1 = math.zeros([1, hiddenSize]);
  let v_w2 = math.zeros([hiddenSize, outputSize]);
  let v_b2 = math.zeros([1, outputSize]);

  let t = 0;

  const relu = (x: any) => math.map(x, (val: number) => Math.max(0, val));
  const reluDeriv = (x: any) => math.map(x, (val: number) => (val > 0 ? 1 : 0));
  
  const softmax = (x: any) => {
    const data = math.matrix(x).toArray() as number[][];
    return data.map(row => {
      const maxVal = Math.max(...row);
      const exps = row.map(v => Math.exp(v - maxVal));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      return exps.map(v => v / (sumExps + 1e-15));
    });
  };

  let metrics: any[] = [];
  let indices = Array.from({ length: trainSamples }, (_, i) => i);

  try {
    for (let epoch = 1; epoch <= epochs; epoch++) {
      const epochStartTime = Date.now();
      
      // Check safety timeout
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

    for (let i = 0; i < trainSamples; i += batchSize) {
      // Heartbeat to prevent watchdog timeout (every 100 batches)
      if (i % (batchSize * 100) === 0) {
        self.postMessage({ type: 'progress', optimizer, epoch, trainProgress: (epoch / epochs) * 100, testProgress: 0 });
      }

      const currentBatchSize = Math.min(batchSize, trainSamples - i);
      const batchIndices = indices.slice(i, i + currentBatchSize);
      
      // Prepare batch data (Vectorized)
      const xBatch = new Array(currentBatchSize);
      const yBatch = new Int32Array(currentBatchSize);
      
      for (let b = 0; b < currentBatchSize; b++) {
        const idx = batchIndices[b];
        // Use subarray directly if possible, or a faster way to get data
        xBatch[b] = Array.from(X_train.subarray(idx * inputSize, (idx + 1) * inputSize));
        yBatch[b] = y_train[idx];
      }

      // Forward Pass (Vectorized)
      const z1 = math.add(math.multiply(xBatch, w1 as any), b1 as any);
      const a1 = relu(z1);
      const z2 = math.add(math.multiply(a1 as any, w2 as any), b2 as any);
      const a2 = softmax(z2);

      // Compute Loss (Cross Entropy)
      let batchLoss = 0;
      const ySmoothed = Array.from({ length: currentBatchSize }, () => new Array(outputSize).fill(0.01 / outputSize));
      yBatch.forEach((label, idx) => {
        ySmoothed[idx][label] = 0.99 + (0.01 / outputSize);
        batchLoss -= Math.log(a2[idx][label] + 1e-15);
      });
      totalLoss += batchLoss / currentBatchSize;

      if (isNaN(totalLoss) || !isFinite(totalLoss)) {
        self.postMessage({ type: 'error', optimizer, message: `Training diverged (Loss is NaN/Infinity) at epoch ${epoch}. Try reducing learning rate.` });
        return;
      }

      // Backward Pass (Vectorized)
      const dz2 = math.divide(math.subtract(a2, ySmoothed), currentBatchSize) as any;
      const dw2 = math.multiply(math.transpose(a1 as any), dz2) as any;
      const db2 = math.reshape(math.sum(dz2, 0), [1, outputSize]) as any;

      const da1 = math.multiply(dz2, math.transpose(w2 as any));
      const dz1 = math.dotMultiply(da1 as any, reluDeriv(z1) as any);
      const dw1 = math.multiply(math.transpose(xBatch as any), dz1 as any) as any;
      const db1 = math.reshape(math.sum(dz1, 0), [1, hiddenSize]) as any;

      // Optimizer Updates
      const eps = 1e-8;
      let u_w1, u_b1, u_w2, u_b2;

      if (optimizer === 'SGD') {
        u_w1 = math.multiply(learningRate, dw1);
        u_b1 = math.multiply(learningRate, db1);
        u_w2 = math.multiply(learningRate, dw2);
        u_b2 = math.multiply(learningRate, db2);
      } else if (optimizer === 'Adagrad') {
        g_w1 = math.add(g_w1 as any, math.dotMultiply(dw1, dw1) as any);
        g_b1 = math.add(g_b1 as any, math.dotMultiply(db1, db1) as any);
        g_w2 = math.add(g_w2 as any, math.dotMultiply(dw2, dw2) as any);
        g_b2 = math.add(g_b2 as any, math.dotMultiply(db2, db2) as any);
        u_w1 = math.dotDivide(math.multiply(learningRate, dw1) as any, math.map(math.add(g_w1 as any, eps) as any, Math.sqrt as any) as any);
        u_b1 = math.dotDivide(math.multiply(learningRate, db1) as any, math.map(math.add(g_b1 as any, eps) as any, Math.sqrt as any) as any);
        u_w2 = math.dotDivide(math.multiply(learningRate, dw2) as any, math.map(math.add(g_w2 as any, eps) as any, Math.sqrt as any) as any);
        u_b2 = math.dotDivide(math.multiply(learningRate, db2) as any, math.map(math.add(g_b2 as any, eps) as any, Math.sqrt as any) as any);
      } else if (optimizer === 'RMSProp') {
        const gamma = 0.9;
        g_w1 = math.add(math.multiply(gamma, g_w1 as any) as any, math.multiply(1 - gamma, math.dotMultiply(dw1, dw1) as any) as any);
        g_b1 = math.add(math.multiply(gamma, g_b1 as any) as any, math.multiply(1 - gamma, math.dotMultiply(db1, db1) as any) as any);
        g_w2 = math.add(math.multiply(gamma, g_w2 as any) as any, math.multiply(1 - gamma, math.dotMultiply(dw2, dw2) as any) as any);
        g_b2 = math.add(math.multiply(gamma, g_b2 as any) as any, math.multiply(1 - gamma, math.dotMultiply(db2, db2) as any) as any);
        u_w1 = math.dotDivide(math.multiply(learningRate, dw1) as any, math.map(math.add(g_w1 as any, eps) as any, Math.sqrt as any) as any);
        u_b1 = math.dotDivide(math.multiply(learningRate, db1) as any, math.map(math.add(g_b1 as any, eps) as any, Math.sqrt as any) as any);
        u_w2 = math.dotDivide(math.multiply(learningRate, dw2) as any, math.map(math.add(g_w2 as any, eps) as any, Math.sqrt as any) as any);
        u_b2 = math.dotDivide(math.multiply(learningRate, db2) as any, math.map(math.add(g_b2 as any, eps) as any, Math.sqrt as any) as any);
      } else { // Adam
        const b1_adam = 0.9, b2_adam = 0.999;
        t++;
        m_w1 = math.add(math.multiply(b1_adam, m_w1 as any) as any, math.multiply(1 - b1_adam, dw1) as any);
        v_w1 = math.add(math.multiply(b2_adam, v_w1 as any) as any, math.multiply(1 - b2_adam, math.dotMultiply(dw1, dw1) as any) as any);
        const mHatW1 = math.divide(m_w1 as any, 1 - Math.pow(b1_adam, t));
        const vHatW1 = math.divide(v_w1 as any, 1 - Math.pow(b2_adam, t));
        u_w1 = math.dotDivide(math.multiply(learningRate, mHatW1 as any) as any, math.add(math.map(vHatW1 as any, Math.sqrt as any) as any, eps) as any);

        m_b1 = math.add(math.multiply(b1_adam, m_b1 as any) as any, math.multiply(1 - b1_adam, db1) as any);
        v_b1 = math.add(math.multiply(b2_adam, v_b1 as any) as any, math.multiply(1 - b2_adam, math.dotMultiply(db1, db1) as any) as any);
        const mHatB1 = math.divide(m_b1 as any, 1 - Math.pow(b1_adam, t));
        const vHatB1 = math.divide(v_b1 as any, 1 - Math.pow(b2_adam, t));
        u_b1 = math.dotDivide(math.multiply(learningRate, mHatB1 as any) as any, math.add(math.map(vHatB1 as any, Math.sqrt as any) as any, eps) as any);

        m_w2 = math.add(math.multiply(b1_adam, m_w2 as any) as any, math.multiply(1 - b1_adam, dw2) as any);
        v_w2 = math.add(math.multiply(b2_adam, v_w2 as any) as any, math.multiply(1 - b2_adam, math.dotMultiply(dw2, dw2) as any) as any);
        const mHatW2 = math.divide(m_w2 as any, 1 - Math.pow(b1_adam, t));
        const vHatW2 = math.divide(v_w2 as any, 1 - Math.pow(b2_adam, t));
        u_w2 = math.dotDivide(math.multiply(learningRate, mHatW2 as any) as any, math.add(math.map(vHatW2 as any, Math.sqrt as any) as any, eps) as any);

        m_b2 = math.add(math.multiply(b1_adam, m_b2 as any) as any, math.multiply(1 - b1_adam, db2) as any);
        v_b2 = math.add(math.multiply(b2_adam, v_b2 as any) as any, math.multiply(1 - b2_adam, math.dotMultiply(db2, db2) as any) as any);
        const mHatB2 = math.divide(m_b2 as any, 1 - Math.pow(b1_adam, t));
        const vHatB2 = math.divide(v_b2 as any, 1 - Math.pow(b2_adam, t));
        u_b2 = math.dotDivide(math.multiply(learningRate, mHatB2 as any) as any, math.add(math.map(vHatB2 as any, Math.sqrt as any) as any, eps) as any);
      }

      // Track Gradient Norm (Corrected: sum of all squared gradients)
      const gradNorm = Math.sqrt(
        (math.sum(math.dotMultiply(dw1, dw1) as any) as any) +
        (math.sum(math.dotMultiply(dw2, dw2) as any) as any) +
        (math.sum(math.dotMultiply(db1, db1) as any) as any) +
        (math.sum(math.dotMultiply(db2, db2) as any) as any)
      );
      totalGradNorm += gradNorm;

      // Track Update Ratio (||deltaW|| / ||W||) (Corrected: sum of all squared updates and weights)
      const updateNorm = Math.sqrt(
        (math.sum(math.dotMultiply(u_w1 as any, u_w1 as any) as any) as any) +
        (math.sum(math.dotMultiply(u_w2 as any, u_w2 as any) as any) as any)
      );
      const weightNorm = Math.sqrt(
        (math.sum(math.dotMultiply(w1 as any, w1 as any) as any) as any) +
        (math.sum(math.dotMultiply(w2 as any, w2 as any) as any) as any)
      );
      totalUpdateNorm += updateNorm / (weightNorm + 1e-12);

      w1 = math.subtract(w1 as any, u_w1 as any);
      b1 = math.subtract(b1 as any, u_b1 as any);
      w2 = math.subtract(w2 as any, u_w2 as any);
      b2 = math.subtract(b2 as any, u_b2 as any);

      batchCount++;
    }

    // Epoch Evaluation
    const avgLoss = totalLoss / batchCount;
    // Fast eval for training accuracy (using TEST data as requested)
    const evalSize = Math.min(1000, testSamples);
    const xEval = new Array(evalSize);
    const yEval = new Array(evalSize);
    for (let j = 0; j < evalSize; j++) {
      const idx = Math.floor(Math.random() * testSamples);
      xEval[j] = Array.from(X_test.subarray(idx * inputSize, (idx + 1) * inputSize));
      yEval[j] = y_test[idx];
    }
    const { a2: a2Eval } = { a2: softmax(math.add(math.multiply(relu(math.add(math.multiply(xEval, w1 as any), b1 as any)), w2 as any), b2 as any)) };
    let correct = 0;
    a2Eval.forEach((pred: any, idx: number) => {
      const predLabel = pred.indexOf(Math.max(...pred));
      if (predLabel === yEval[idx]) correct++;
    });
    const accuracy = correct / evalSize;

    const metric = {
      epoch,
      loss: avgLoss,
      accuracy,
      gradientNorm: totalGradNorm / batchCount,
      updateRatio: totalUpdateNorm / batchCount,
      convergenceSpeed: metrics.length > 0 ? metrics[metrics.length - 1].loss - avgLoss : 0
    };
    metrics.push(metric);

    // Benchmark (Step 8)
    const epochDuration = Date.now() - epochStartTime;
    if (epochDuration > 2000) { // 2 seconds threshold for warning
      console.warn(`[${optimizer}] Epoch ${epoch} took ${epochDuration}ms. Consider increasing batch size or reducing hidden layer size.`);
    }

    self.postMessage({ 
      type: 'progress', 
      optimizer, 
      epoch, 
      metric, 
      trainProgress: (epoch / epochs) * 100,
      testProgress: 0
    });

    // Asynchronous pause to keep worker responsive
    await new Promise(resolve => setTimeout(resolve, 0));
  }

  // Final Testing (Step 2 - Step 11)
  const testStartTime = Date.now();
  const y_true: number[] = [];
  const y_pred: number[] = [];
  const y_probs: number[][] = [];
  
  const testBatchSize = 500;
  for (let i = 0; i < testSamples; i += testBatchSize) {
    const currentBatchSize = Math.min(testBatchSize, testSamples - i);
    const xBatch = new Array(currentBatchSize);
    const yBatch = new Array(currentBatchSize);
    for (let b = 0; b < currentBatchSize; b++) {
      const idx = i + b;
      xBatch[b] = Array.from(X_test.subarray(idx * inputSize, (idx + 1) * inputSize));
      yBatch[b] = y_test[idx];
    }

    const { a2 } = { a2: softmax(math.add(math.multiply(relu(math.add(math.multiply(xBatch, w1 as any), b1 as any)), w2 as any), b2 as any)) };
    a2.forEach((pred: any, idx: number) => {
      const predLabel = pred.indexOf(Math.max(...pred));
      const trueLabel = yBatch[idx];
      y_true.push(trueLabel);
      y_pred.push(predLabel);
      y_probs.push(pred);
    });

    self.postMessage({
      type: 'progress',
      optimizer,
      epoch: epochs,
      trainProgress: 100,
      testProgress: Math.min(100, ((i + currentBatchSize) / testSamples) * 100)
    });
  }

  // Step 3: Build Confusion Matrix with defensive sizing
  const maxTrueLabel = y_true.length > 0 ? Math.max(...y_true) : 0;
  const maxPredLabel = y_pred.length > 0 ? Math.max(...y_pred) : 0;
  const actualOutputSize = Math.max(outputSize, maxTrueLabel + 1, maxPredLabel + 1);
  
  const confusionMatrix = Array.from({ length: actualOutputSize }, () => new Array(actualOutputSize).fill(0));
  for (let i = 0; i < testSamples; i++) {
    const t = y_true[i];
    const p = y_pred[i];
    if (t >= 0 && t < actualOutputSize && p >= 0 && p < actualOutputSize) {
      confusionMatrix[t][p]++;
    }
  }

  // Step 4: Compute Accuracy
  let totalCorrect = 0;
  for (let i = 0; i < actualOutputSize; i++) {
    totalCorrect += confusionMatrix[i][i];
  }
  const testAccuracy = totalCorrect / testSamples;

  // Step 5 & 6: Compute Per-Class and Weighted Metrics
  let weightedPrecision = 0;
  let weightedRecall = 0;
  let weightedF1 = 0;
  let totalSupport = 0;

  for (let i = 0; i < actualOutputSize; i++) {
    // tp: True Positives for class i
    const tp = confusionMatrix[i][i];
    
    // support: Total samples that are actually class i (row sum)
    let support = 0;
    for (let j = 0; j < actualOutputSize; j++) support += confusionMatrix[i][j];
    
    // totalPredicted: Total times the model predicted class i (column sum)
    let totalPredicted = 0;
    for (let j = 0; j < actualOutputSize; j++) totalPredicted += confusionMatrix[j][i];

    // We only calculate metrics for classes that actually exist in the test set
    if (support > 0) {
      const p = totalPredicted > 0 ? tp / totalPredicted : 0;
      const r = tp / support; 
      const f = (p + r) > 0 ? (2 * p * r) / (p + r) : 0;
      
      weightedPrecision += p * support;
      weightedRecall += r * support;
      weightedF1 += f * support;
      totalSupport += support;
    }
  }

  // Weighted averaging: sum of (metric * support) / total support
  const precision = totalSupport > 0 ? weightedPrecision / totalSupport : 0;
  const recall = totalSupport > 0 ? weightedRecall / totalSupport : 0;
  const f1Score = totalSupport > 0 ? weightedF1 / totalSupport : 0;

  // Step 7: Compute Log Loss
  let totalLogLoss = 0;
  for (let i = 0; i < testSamples; i++) {
    const trueLabel = y_true[i];
    const prob = y_probs[i][trueLabel];
    totalLogLoss -= Math.log(prob + 1e-15);
  }
  const logLoss = totalLogLoss / testSamples;

  // Step 8: Compute Convergence Speed (Corrected formula)
  const convergenceRate =
    metrics.length > 1
      ? (metrics[0].loss - metrics[metrics.length - 1].loss) /
        (metrics.length * Math.max(...metrics.map(m => m.loss)))
      : 0;

  // Step 9: Compute Loss Variance
  const avgLoss = metrics.reduce((sum, m) => sum + m.loss, 0) / metrics.length;
  const lossVariance = metrics.reduce((sum, m) => sum + Math.pow(m.loss - avgLoss, 2), 0) / metrics.length;

  // Step 10: Record Execution Time
  const trainingTime = (testStartTime - startTime) / 1000;
  const testingTime = (Date.now() - testStartTime) / 1000;

  // Step 11: Store Final Metrics
  self.postMessage({
    type: 'training_complete',
    optimizer,
    metrics: {
      optimizer,
      metrics,
      testAccuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix,
      logLoss,
      trainingTime,
      testingTime,
      executionTime: trainingTime + testingTime,
      convergenceRate,
      aulc: metrics.reduce((acc, m, i) => {
        if (i === 0) return acc;
        return acc + (metrics[i-1].accuracy + m.accuracy) / 2;
      }, 0),
      lossVariance
    }
  });
  } catch (err: any) {
    self.postMessage({ type: 'error', optimizer, message: err.message || 'Unknown worker error' });
  } finally {
    // Memory Cleanup
    (indices as any) = null;
    (metrics as any) = null;
  }
};
