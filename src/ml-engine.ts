import * as math from 'mathjs';

export type OptimizerType = 'SGD' | 'Adagrad' | 'RMSProp' | 'Adam';

export interface ModelParams {
  hiddenSize: number;
  learningRate: number;
  epochs: number;
  batchSize: number;
}

export interface TrainingMetric {
  epoch: number;
  loss: number;
  accuracy: number;
  gradientNorm: number;
  updateRatio: number;
  convergenceSpeed: number;
}

export interface ExperimentResult {
  optimizer: OptimizerType;
  metrics: TrainingMetric[];
  testAccuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: number[][];
  logLoss: number;
  executionTime: number;
  convergenceRate: number;
  lossVariance: number;
  aulc: number;
}

export class NeuralNetwork {
  private w1: any;
  private b1: any;
  private w2: any;
  private b2: any;
  
  // Optimizer states
  private g_w1: any; private g_b1: any; private g_w2: any; private g_b2: any; // Adagrad/RMSProp
  private m_w1: any; private m_b1: any; private m_w2: any; private m_b2: any; // Adam momentum
  private v_w1: any; private v_b1: any; private v_w2: any; private v_b2: any; // Adam velocity
  private t: number = 0;

  constructor(inputSize: number, hiddenSize: number, outputSize: number) {
    // He Initialization (better for ReLU)
    const scale1 = Math.sqrt(2.0 / inputSize);
    const scale2 = Math.sqrt(2.0 / hiddenSize);
    
    this.w1 = math.multiply(math.random([inputSize, hiddenSize], -1, 1), scale1);
    this.b1 = math.zeros([1, hiddenSize]);
    this.w2 = math.multiply(math.random([hiddenSize, outputSize], -1, 1), scale2);
    this.b2 = math.zeros([1, outputSize]);

    // Init optimizer states
    this.g_w1 = math.zeros([inputSize, hiddenSize]);
    this.g_b1 = math.zeros([1, hiddenSize]);
    this.g_w2 = math.zeros([hiddenSize, outputSize]);
    this.g_b2 = math.zeros([1, outputSize]);

    this.m_w1 = math.zeros([inputSize, hiddenSize]);
    this.m_b1 = math.zeros([1, hiddenSize]);
    this.m_w2 = math.zeros([hiddenSize, outputSize]);
    this.m_b2 = math.zeros([1, outputSize]);

    this.v_w1 = math.zeros([inputSize, hiddenSize]);
    this.v_b1 = math.zeros([1, hiddenSize]);
    this.v_w2 = math.zeros([hiddenSize, outputSize]);
    this.v_b2 = math.zeros([1, outputSize]);
  }

  private relu(x: any) {
    return math.map(x, (val: number) => Math.max(0, val));
  }

  private reluDeriv(x: any) {
    return math.map(x, (val: number) => (val > 0 ? 1 : 0));
  }

  private softmax(x: any) {
    // x is [batch, outputSize]
    const data = math.matrix(x).toArray() as number[][];
    return data.map(row => {
      const maxVal = Math.max(...row);
      const exps = row.map(v => Math.exp(v - maxVal));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      return exps.map(v => v / (sumExps + 1e-15));
    });
  }

  forward(x: any) {
    const z1 = math.add(math.multiply(x, this.w1), this.b1);
    const a1 = this.relu(z1);
    const z2 = math.add(math.multiply(a1, this.w2), this.b2);
    const a2 = this.softmax(z2);
    return { z1, a1, z2, a2 };
  }

  computeLoss(yPred: any, yTrue: any) {
    // Cross entropy
    const batchSize = yTrue.length;
    let loss = 0;
    for (let i = 0; i < batchSize; i++) {
      const label = yTrue[i];
      loss -= Math.log(yPred[i][label] + 1e-15);
    }
    return loss / batchSize;
  }

  trainStep(x: any, y: any, lr: number, optimizer: OptimizerType) {
    const batchSize = x.length;
    const { z1, a1, z2, a2 } = this.forward(x);

    // Label smoothing
    const numClasses = a2[0].length;
    const smoothing = 0.1;
    const ySmoothed = Array.from({ length: batchSize }, () => new Array(numClasses).fill(smoothing / numClasses));
    y.forEach((label: number, i: number) => {
      if (ySmoothed[i]) {
        ySmoothed[i][label] = (1 - smoothing) + (smoothing / numClasses);
      }
    });

    // Backprop
    const dz2 = math.divide(math.subtract(a2, ySmoothed), batchSize);
    const dw2 = math.add(math.multiply(math.transpose(a1), dz2), math.multiply(0.001, this.w2)); // Increased L2 Regularization
    const db2 = math.multiply(math.ones([1, batchSize]), dz2) as any;

    const da1 = math.multiply(dz2, math.transpose(this.w2));
    const dz1 = math.dotMultiply(da1, this.reluDeriv(z1));
    const dw1 = math.add(math.multiply(math.transpose(x), dz1), math.multiply(0.001, this.w1)); // Increased L2 Regularization
    const db1 = math.multiply(math.ones([1, batchSize]), dz1) as any;

    const grads = { dw1, db1, dw2, db2 };
    const { updateNorm } = this.updateWeights(grads, lr, optimizer);

    // Metrics for dashboard
    const gradNorm = Math.sqrt(
      ((math as any).sum((math as any).dotMultiply(dw1, dw1)) as any) + ((math as any).sum((math as any).dotMultiply(dw2, dw2)) as any)
    );
    
    return { gradNorm, updateNorm };
  }

  private updateWeights(grads: any, lr: number, optimizer: OptimizerType) {
    const { dw1, db1, dw2, db2 } = grads;
    const eps = 1e-8;
    let u_w1, u_b1, u_w2, u_b2;

    if (optimizer === 'SGD') {
      u_w1 = math.multiply(lr, dw1);
      u_b1 = math.multiply(lr, db1);
      u_w2 = math.multiply(lr, dw2);
      u_b2 = math.multiply(lr, db2);
    } else if (optimizer === 'Adagrad') {
      this.g_w1 = math.add(this.g_w1, (math as any).dotMultiply(dw1, dw1));
      this.g_b1 = math.add(this.g_b1, (math as any).dotMultiply(db1, db1));
      this.g_w2 = math.add(this.g_w2, (math as any).dotMultiply(dw2, dw2));
      this.g_b2 = math.add(this.g_b2, (math as any).dotMultiply(db2, db2));

      u_w1 = math.dotDivide(math.multiply(lr, dw1), (math as any).map(math.add(this.g_w1, eps), math.sqrt) as any);
      u_b1 = math.dotDivide(math.multiply(lr, db1), (math as any).map(math.add(this.g_b1, eps), math.sqrt) as any);
      u_w2 = math.dotDivide(math.multiply(lr, dw2), (math as any).map(math.add(this.g_w2, eps), math.sqrt) as any);
      u_b2 = math.dotDivide(math.multiply(lr, db2), (math as any).map(math.add(this.g_b2, eps), math.sqrt) as any);
    } else if (optimizer === 'RMSProp') {
      const gamma = 0.9;
      this.g_w1 = math.add(math.multiply(gamma, this.g_w1), math.multiply(1 - gamma, (math as any).dotMultiply(dw1, dw1)));
      this.g_b1 = math.add(math.multiply(gamma, this.g_b1), math.multiply(1 - gamma, (math as any).dotMultiply(db1, db1)));
      this.g_w2 = math.add(math.multiply(gamma, this.g_w2), math.multiply(1 - gamma, (math as any).dotMultiply(dw2, dw2)));
      this.g_b2 = math.add(math.multiply(gamma, this.g_b2), math.multiply(1 - gamma, (math as any).dotMultiply(db2, db2)));

      u_w1 = math.dotDivide(math.multiply(lr, dw1), (math as any).map(math.add(this.g_w1, eps), math.sqrt) as any);
      u_b1 = math.dotDivide(math.multiply(lr, db1), (math as any).map(math.add(this.g_b1, eps), math.sqrt) as any);
      u_w2 = math.dotDivide(math.multiply(lr, dw2), (math as any).map(math.add(this.g_w2, eps), math.sqrt) as any);
      u_b2 = math.dotDivide(math.multiply(lr, db2), (math as any).map(math.add(this.g_b2, eps), math.sqrt) as any);
    } else { // Adam
      const b1 = 0.9, b2 = 0.999;
      this.t++;
      
      this.m_w1 = math.add(math.multiply(b1, this.m_w1), math.multiply(1 - b1, dw1));
      this.v_w1 = math.add(math.multiply(b2, this.v_w1), math.multiply(1 - b2, (math as any).dotMultiply(dw1, dw1)));
      const mHatW1 = math.divide(this.m_w1, 1 - Math.pow(b1, this.t));
      const vHatW1 = math.divide(this.v_w1, 1 - Math.pow(b2, this.t));
      u_w1 = math.dotDivide(math.multiply(lr, mHatW1), math.add((math as any).map(vHatW1, math.sqrt), eps) as any);

      this.m_b1 = math.add(math.multiply(b1, this.m_b1), math.multiply(1 - b1, db1));
      this.v_b1 = math.add(math.multiply(b2, this.v_b1), math.multiply(1 - b2, (math as any).dotMultiply(db1, db1)));
      const mHatB1 = math.divide(this.m_b1, 1 - Math.pow(b1, this.t));
      const vHatB1 = math.divide(this.v_b1, 1 - Math.pow(b2, this.t));
      u_b1 = math.dotDivide(math.multiply(lr, mHatB1), math.add((math as any).map(vHatB1, math.sqrt), eps) as any);

      this.m_w2 = math.add(math.multiply(b1, this.m_w2), math.multiply(1 - b1, dw2));
      this.v_w2 = math.add(math.multiply(b2, this.v_w2), math.multiply(1 - b2, (math as any).dotMultiply(dw2, dw2)));
      const mHatW2 = math.divide(this.m_w2, 1 - Math.pow(b1, this.t));
      const vHatW2 = math.divide(this.v_w2, 1 - Math.pow(b2, this.t));
      u_w2 = math.dotDivide(math.multiply(lr, mHatW2), math.add((math as any).map(vHatW2, math.sqrt), eps) as any);

      this.m_b2 = math.add(math.multiply(b1, this.m_b2), math.multiply(1 - b1, db2));
      this.v_b2 = math.add(math.multiply(b2, this.v_b2), math.multiply(1 - b2, (math as any).dotMultiply(db2, db2)));
      const mHatB2 = math.divide(this.m_b2, 1 - Math.pow(b1, this.t));
      const vHatB2 = math.divide(this.v_b2, 1 - Math.pow(b2, this.t));
      u_b2 = math.dotDivide(math.multiply(lr, mHatB2), math.add((math as any).map(vHatB2, math.sqrt), eps) as any);
    }

    this.w1 = math.subtract(this.w1, u_w1);
    this.b1 = math.subtract(this.b1, u_b1);
    this.w2 = math.subtract(this.w2, u_w2);
    this.b2 = math.subtract(this.b2, u_b2);

    const updateNorm = Math.sqrt(
      ((math as any).sum((math as any).dotMultiply(u_w1, u_w1)) as any) + ((math as any).sum((math as any).dotMultiply(u_w2, u_w2)) as any)
    );

    return { updateNorm };
  }

  evaluate(x: any, y: any) {
    const { a2 } = this.forward(x);
    const numClasses = a2[0].length;
    const confusionMatrix = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
    let correct = 0;
    let logLoss = 0;

    a2.forEach((pred: any, i: number) => {
      const predLabel = pred.indexOf(Math.max(...pred));
      const trueLabel = y[i];
      confusionMatrix[trueLabel][predLabel]++;
      if (predLabel === trueLabel) correct++;
      logLoss -= Math.log(pred[trueLabel] + 1e-15);
    });

    const accuracy = correct / y.length;
    logLoss /= y.length;

    // Calculate Macro Precision, Recall, F1 (Average of per-class metrics)
    let totalPrecision = 0;
    let totalRecall = 0;
    let totalF1 = 0;

    for (let i = 0; i < numClasses; i++) {
      const tp = confusionMatrix[i][i];
      const fp = confusionMatrix.reduce((sum, row, idx) => (idx !== i ? sum + row[i] : sum), 0);
      const fn = confusionMatrix[i].reduce((sum, val, idx) => (idx !== i ? sum + val : sum), 0);

      const p = tp + fp > 0 ? tp / (tp + fp) : 0;
      const r = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f = p + r > 0 ? (2 * p * r) / (p + r) : 0;

      totalPrecision += p;
      totalRecall += r;
      totalF1 += f;
    }

    const precision = totalPrecision / numClasses;
    const recall = totalRecall / numClasses;
    const f1Score = (precision + recall > 0) ? (2 * precision * recall) / (precision + recall) : 0;

    return {
      accuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix,
      logLoss
    };
  }
}
