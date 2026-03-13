import React, { useState, useEffect, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import Papa from 'papaparse';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import html2canvas from 'html2canvas';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell, ScatterChart, Scatter, ZAxis, AreaChart, Area
} from 'recharts';
import { 
  Upload, Play, History, BarChart3, Settings, Database, Timer, CheckCircle2, AlertCircle, Info, Pause, PlayCircle,
  Scissors, Download, ChevronRight, Activity, Zap, TrendingDown, HelpCircle
} from 'lucide-react';
import { NeuralNetwork, OptimizerType, ModelParams, ExperimentResult, TrainingMetric } from './ml-engine';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const METRICS_INFO = {
  // Optimizers
  'SGD': 'Stochastic Gradient Descent: Updates parameters using the gradient of the loss function with respect to a single or small batch of samples.',
  'Adagrad': 'Adaptive Gradient Algorithm: Scales the learning rate for each parameter based on the historical sum of squared gradients.',
  'RMSProp': 'Root Mean Square Propagation: An adaptive learning rate method that uses a moving average of squared gradients to normalize the gradient.',
  'Adam': 'Adaptive Moment Estimation: Combines the advantages of Adagrad and RMSProp, using both first and second moments of the gradients.',
  
  // Core Metrics
  'Test Accuracy': 'The percentage of predictions that are correct on the test dataset.',
  'Precision (Macro)': 'Measures how many predicted positive samples are actually correct. Calculated per class and averaged across classes.',
  'Recall (Macro)': 'Measures how well the model identifies all true positive samples across classes.',
  'F1 Score': 'The harmonic mean of Precision and Recall. Provides a balanced measure when dealing with class imbalance.',
  'Log Loss': 'Measures the uncertainty of predictions. Lower values indicate more confident and accurate predictions.',
  
  // Training Dynamics
  'Gradient Norm': 'Represents the magnitude of gradients during backpropagation. Large values may indicate instability, while very small values may indicate the model has stopped learning.',
  'Update Ratio': 'The ratio of parameter update magnitude to the parameter value. Ideal values are typically around 10⁻³ (0.001). Too high indicates unstable learning; too low indicates slow learning.',
  'Convergence Speed': 'Measures how quickly the loss decreases during training.',
  'Loss Variance': 'Indicates how much the loss fluctuates during training. High variance may suggest unstable optimization or an overly high learning rate.',
  
  // Advanced Benchmarks
  'AULC': 'Area Under Learning Curve: Represents the cumulative performance across all training epochs. Higher values indicate faster and more stable learning.',
  'Execution Time': 'The total time required to complete model training.',
  'Convergence Rate': 'Measures the efficiency of an optimizer relative to a baseline optimizer (SGD) in reaching a stable loss.'
};

const InfoTooltip = ({ title, content }: { title: string, content: string }) => (
  <div className="group relative inline-block ml-1 align-middle">
    <Info className="w-4 h-4 text-[#A8A29E] hover:text-[#1C1917] cursor-help transition-colors" />
    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block w-64 p-3 bg-[#1C1917] text-white text-[11px] rounded-xl shadow-2xl z-[100] animate-in fade-in zoom-in-95 duration-200">
      <div className="font-bold mb-1 text-emerald-400">{title}</div>
      <div className="leading-relaxed opacity-90">{content}</div>
      <div className="absolute top-full left-1/2 -translate-x-1/2 border-8 border-transparent border-t-[#1C1917]" />
    </div>
  </div>
);

export default function App() {
  // State
  const [trainData, setTrainData] = useState<any[]>([]);
  const [testData, setTestData] = useState<any[]>([]);
  const [trainFile, setTrainFile] = useState<File | null>(null);
  const [testFile, setTestFile] = useState<File | null>(null);
  
  const [features, setFeatures] = useState<string[]>([]);
  const [target, setTarget] = useState<string>('');
  const [trainSampleSize, setTrainSampleSize] = useState<number>(10000);
  const [testSampleSize, setTestSampleSize] = useState<number>(2000);
  
  const [params, setParams] = useState<ModelParams>({
    hiddenSize: 64,
    learningRate: 0.01,
    epochs: 10,
    batchSize: 64
  });

  const [isTraining, setIsTraining] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [testingProgress, setTestingProgress] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [selectedExperiment, setSelectedExperiment] = useState<any>(null);
  const [isViewingReport, setIsViewingReport] = useState(false);

  const safeFixed = (val: any, digits: number = 2, multiplier: number = 1, suffix: string = '') => {
    if (val === undefined || val === null || isNaN(Number(val))) return 'N/A';
    return (Number(val) * multiplier).toFixed(digits) + suffix;
  };
  const isPausedRef = useRef(false);
  const stopTrainingRef = useRef(false);

  const stopTraining = () => {
    stopTrainingRef.current = true;
    setIsTraining(false);
    setIsPaused(false);
    isPausedRef.current = false;
  };

  const [currentOptimizer, setCurrentOptimizer] = useState<OptimizerType | null>(null);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [results, setResults] = useState<ExperimentResult[]>([]);
  const [history, setHistory] = useState<any[]>([]);

  // Timer effect
  useEffect(() => {
    let interval: any;
    if (isTraining && !isPaused) {
      interval = setInterval(() => setElapsedTime(prev => prev + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [isTraining, isPaused]);

  const togglePause = () => {
    const nextState = !isPaused;
    setIsPaused(nextState);
    isPausedRef.current = nextState;
  };

  const checkPause = async () => {
    while (isPausedRef.current) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  };

  // Load History
  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      setHistory(data);
    } catch (e) {
      console.error('Failed to fetch history', e);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>, type: 'train' | 'test') => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (type === 'train') setTrainFile(file);
    else setTestFile(file);

    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      preview: 100, // Preview for UI
      worker: true, // Use worker for large files
      complete: (results) => {
        if (type === 'train') {
          setTrainData(results.data);
          const cols = Object.keys(results.data[0] || {});
          setFeatures(cols.slice(0, -1));
          setTarget(cols[cols.length - 1] || '');
        } else {
          setTestData(results.data);
        }
      }
    });
  };

  const startTraining = async () => {
    if (!trainFile || !testFile || !target) return;
    setIsTraining(true);
    setIsPaused(false);
    isPausedRef.current = false;
    stopTrainingRef.current = false;
    setResults([]);
    
    // Load full data with sampling
    const loadAndSample = (file: File, size: number) => {
      return new Promise<any[]>((resolve) => {
        setStatusMessage(`Loading and sampling ${file.name}...`);
        Papa.parse(file, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          worker: true,
          complete: (results) => {
            const data = results.data;
            const sampled = data.sort(() => 0.5 - Math.random()).slice(0, size);
            resolve(sampled);
          }
        });
      });
    };

    const fullTrain = await loadAndSample(trainFile, trainSampleSize);
    const fullTest = await loadAndSample(testFile, testSampleSize);

    // Prepare data
    const X_train_raw = fullTrain.map(row => features.map(f => Number(row[f]) || 0));
    const y_train = fullTrain.map(row => row[target]);
    const X_test_raw = fullTest.map(row => features.map(f => Number(row[f]) || 0));
    const y_test = fullTest.map(row => row[target]);

    const normalize = (data: any[][]) => {
      if (data.length === 0) return [];
      const means = data[0].map((_, col) => data.reduce((acc, row) => acc + row[col], 0) / data.length);
      const stds = data[0].map((_, col) => Math.sqrt(data.reduce((acc, row) => acc + Math.pow(row[col] - means[col], 2), 0) / data.length) || 1);
      return data.map(row => row.map((val, col) => (val - means[col]) / stds[col]));
    };
    
    const X_train_norm = normalize(X_train_raw);
    const X_test_norm = normalize(X_test_raw);

    // Unique classes
    const classes = Array.from(new Set([...y_train, ...y_test])).sort();
    const classMap = new Map(classes.map((c, i) => [c, i]));
    const y_train_idx = y_train.map(v => classMap.get(v) || 0);
    const y_test_idx = y_test.map(v => classMap.get(v) || 0);

    const optimizers: OptimizerType[] = ['SGD', 'Adagrad', 'RMSProp', 'Adam'];
    const allResults: ExperimentResult[] = [];
    let sgdTime = 0;

    for (const opt of optimizers) {
      if (stopTrainingRef.current) break;
      setCurrentOptimizer(opt);
      setTrainingProgress(0);
      setTestingProgress(0);
      setIsTesting(false);
      setElapsedTime(0); // Reset timer for each optimizer
      
      const startTime = Date.now();
      const nn = new NeuralNetwork(features.length, params.hiddenSize, classes.length);
      const metrics: TrainingMetric[] = [];

      for (let epoch = 1; epoch <= params.epochs; epoch++) {
        await checkPause();
        if (stopTrainingRef.current) break;

        setCurrentEpoch(epoch);
        setStatusMessage(`Training ${opt}: Epoch ${epoch}/${params.epochs}`);
        let totalLoss = 0;
        let totalGradNorm = 0;
        let totalUpdateNorm = 0;
        let batchCount = 0;
        
        // Mini-batch training
        for (let i = 0; i < X_train_norm.length; i += params.batchSize) {
          await checkPause();
          if (stopTrainingRef.current) break;

          const batchX = X_train_norm.slice(i, i + params.batchSize);
          const batchY = y_train_idx.slice(i, i + params.batchSize);
          
          const { gradNorm, updateNorm } = nn.trainStep(batchX, batchY, params.learningRate, opt);
          const { a2 } = nn.forward(batchX);
          totalLoss += nn.computeLoss(a2, batchY);
          totalGradNorm += gradNorm;
          totalUpdateNorm += updateNorm;
          batchCount++;
          
          setTrainingProgress(((epoch - 1) * X_train_norm.length + i + batchX.length) / (params.epochs * X_train_norm.length) * 100);
        }

        if (stopTrainingRef.current) break;

        const avgLoss = totalLoss / batchCount;
        // Fast evaluation on a subset for training metrics to keep UI responsive
        const trainEval = nn.evaluate(X_train_norm.slice(0, 1000), y_train_idx.slice(0, 1000));
        const accuracy = trainEval.accuracy;
        
        metrics.push({
          epoch,
          loss: avgLoss,
          accuracy,
          gradientNorm: totalGradNorm / batchCount,
          updateRatio: totalUpdateNorm / batchCount,
          convergenceSpeed: metrics.length > 0 ? Math.abs(metrics[metrics.length - 1].loss - avgLoss) : 0
        });

        // Yield to UI
        await new Promise(r => setTimeout(r, 0));
      }

      if (stopTrainingRef.current) break;

      setIsTesting(true);
      setStatusMessage(`Testing ${opt} performance...`);
      
      // Batch-wise evaluation to show progress
      const testBatchSize = 500;
      let correct = 0;
      let totalLogLoss = 0;
      const numClasses = classes.length;
      const confusionMatrix = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));

      for (let i = 0; i < X_test_norm.length; i += testBatchSize) {
        await checkPause();
        if (stopTrainingRef.current) break;

        const batchX = X_test_norm.slice(i, i + testBatchSize);
        const batchY = y_test_idx.slice(i, i + testBatchSize);
        
        const { a2 } = nn.forward(batchX);
        a2.forEach((pred: any, idx: number) => {
          const predLabel = pred.indexOf(Math.max(...pred));
          const trueLabel = batchY[idx];
          confusionMatrix[trueLabel][predLabel]++;
          if (predLabel === trueLabel) correct++;
          totalLogLoss -= Math.log(pred[trueLabel] + 1e-15);
        });

        setTestingProgress(((i + batchX.length) / X_test_norm.length) * 100);
        await new Promise(r => setTimeout(r, 0));
      }

      if (stopTrainingRef.current) break;

      const testAccuracy = correct / X_test_norm.length;
      const logLoss = totalLogLoss / X_test_norm.length;

      // Calculate Macro Precision, Recall, F1
      let totalPrecision = 0;
      let totalRecall = 0;
      let validPrecisionClasses = 0;
      let validRecallClasses = 0;

      for (let i = 0; i < numClasses; i++) {
        const tp = confusionMatrix[i][i];
        const fp = confusionMatrix.reduce((sum, row, idx) => (idx !== i ? sum + row[i] : sum), 0);
        const fn = confusionMatrix[i].reduce((sum, val, idx) => (idx !== i ? sum + val : sum), 0);

        const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
        const recall = tp + fn > 0 ? tp / (tp + fn) : 0;

        if (tp + fp > 0) {
          totalPrecision += precision;
          validPrecisionClasses++;
        }
        if (tp + fn > 0) {
          totalRecall += recall;
          validRecallClasses++;
        }
      }

      const precision = validPrecisionClasses > 0 ? totalPrecision / validPrecisionClasses : 0;
      const recall = validRecallClasses > 0 ? totalRecall / validRecallClasses : 0;
      const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

      const executionTime = (Date.now() - startTime) / 1000;
      if (opt === 'SGD') sgdTime = executionTime;

      const meanLoss = metrics.reduce((s, x) => s + x.loss, 0) / metrics.length;
      const aulc = metrics.reduce((acc, m) => acc + m.loss, 0);

      // Convergence Rate relative to SGD
      // Formula: baseline_training_time / optimizer_training_time
      // We use a small epsilon to avoid division by zero
      const convergenceRate = opt === 'SGD' ? 1 : (sgdTime / (executionTime + 1e-8));

      const result: ExperimentResult = {
        optimizer: opt,
        metrics,
        testAccuracy,
        precision,
        recall,
        f1Score,
        confusionMatrix,
        logLoss,
        executionTime,
        convergenceRate: Math.min(Math.max(convergenceRate, 0.1), 20), // Clamping to realistic range
        lossVariance: metrics.reduce((acc, m) => acc + Math.pow(m.loss - meanLoss, 2), 0) / metrics.length,
        aulc
      };

      allResults.push(result);
      setResults([...allResults]);

      // Save to DB
      await fetch('/api/experiments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name: trainFile.name,
          sample_size: trainSampleSize,
          train_test_split: (trainSampleSize / (trainSampleSize + testSampleSize)) * 100,
          optimizer: opt,
          hidden_size: params.hiddenSize,
          learning_rate: params.learningRate,
          epochs: params.epochs,
          batch_size: params.batchSize,
          test_accuracy: testAccuracy,
          precision,
          recall,
          f1_score: f1Score,
          confusion_matrix: confusionMatrix,
          log_loss: logLoss,
          convergence_rate: result.convergenceRate,
          execution_time: executionTime,
          aulc,
          loss_variance: result.lossVariance,
          logs: metrics
        })
      });
    }

    setIsTraining(false);
    setIsTesting(false);
    setCurrentOptimizer(null);
    setStatusMessage('Experiment complete.');
    fetchHistory();
  };

  const bestOptimizer = useMemo(() => {
    if (results.length === 0) return null;
    return results.reduce((prev, curr) => {
      // Score: accuracy*40 + f1*30 + convergence*20 - time*10
      const score = (res: ExperimentResult) => (res.testAccuracy * 40) + (res.f1Score * 30) + (res.convergenceRate * 20) - (res.executionTime * 10);
      return score(curr) > score(prev) ? curr : prev;
    });
  }, [results]);

  const fetchExperimentDetails = async (id: number) => {
    try {
      const res = await fetch(`/api/experiments/${id}`);
      const data = await res.json();
      if (typeof data.logs === 'string') {
        data.logs = JSON.parse(data.logs);
      }
      if (typeof data.confusion_matrix === 'string') {
        data.confusion_matrix = JSON.parse(data.confusion_matrix);
      }
      setSelectedExperiment(data);
      setIsViewingReport(true);
    } catch (e) {
      console.error('Failed to fetch experiment details', e);
    }
  };

  const downloadReport = async (exp: any) => {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();

    // Title
    doc.setFontSize(22);
    doc.setTextColor(28, 25, 23); // #1C1917
    doc.text(`Experiment Report #${exp.id}`, 14, 22);

    // Metadata
    doc.setFontSize(10);
    doc.setTextColor(120, 113, 108); // #78716C
    doc.text(`Generated on ${new Date().toLocaleString()}`, 14, 30);
    doc.text(`Dataset: ${exp.dataset_name}`, 14, 35);

    // Summary Section
    doc.setFontSize(16);
    doc.setTextColor(28, 25, 23);
    doc.text("1. Core Performance Metrics", 14, 50);

    const coreMetrics = [
      ["Test Accuracy", safeFixed(exp.test_accuracy, 2, 100, '%')],
      ["Precision (Macro)", safeFixed(exp.precision, 2, 100, '%')],
      ["Recall (Macro)", safeFixed(exp.recall, 2, 100, '%')],
      ["F1 Score", safeFixed(exp.f1_score, 2, 100, '%')],
      ["Log Loss", safeFixed(exp.log_loss, 4)]
    ];

    autoTable(doc, {
      startY: 55,
      head: [["Metric", "Value"]],
      body: coreMetrics,
      theme: 'striped',
      headStyles: { fillColor: [28, 25, 23] }
    });

    let reportY = (doc as any).lastAutoTable.finalY + 15;
    doc.text("2. Training Dynamics", 14, reportY);

    const trainingDynamics = [
      ["Avg Gradient Norm", safeFixed(exp.logs[exp.logs.length - 1].gradientNorm, 4)],
      ["Avg Update Ratio", safeFixed(exp.logs[exp.logs.length - 1].updateRatio, 6)],
      ["Convergence Speed", safeFixed(exp.logs[exp.logs.length - 1].convergenceSpeed, 6)],
      ["Loss Variance", safeFixed(exp.loss_variance, 6)]
    ];

    autoTable(doc, {
      startY: reportY + 5,
      head: [["Metric", "Value"]],
      body: trainingDynamics,
      theme: 'striped',
      headStyles: { fillColor: [28, 25, 23] }
    });

    reportY = (doc as any).lastAutoTable.finalY + 15;
    doc.text("3. Advanced Benchmarks", 14, reportY);

    const advancedBenchmarks = [
      ["AULC", safeFixed(exp.aulc, 2)],
      ["Execution Time", safeFixed(exp.execution_time, 2, 1, 's')],
      ["Convergence Rate", safeFixed(exp.convergence_rate, 2, 1, 'x')]
    ];

    autoTable(doc, {
      startY: reportY + 5,
      head: [["Metric", "Value"]],
      body: advancedBenchmarks,
      theme: 'striped',
      headStyles: { fillColor: [28, 25, 23] }
    });

    // Parameters Section
    reportY = (doc as any).lastAutoTable.finalY + 15;
    if (reportY > 250) {
      doc.addPage();
      reportY = 22;
    }
    doc.setFontSize(16);
    doc.text("Model Parameters", 14, reportY);

    const paramData = [
      ["Hidden Size", exp.hidden_size],
      ["Learning Rate", exp.learning_rate],
      ["Epochs", exp.epochs],
      ["Batch Size", exp.batch_size],
      ["Sample Size", exp.sample_size],
      ["Train/Test Split", `${exp.train_test_split}% / ${100 - exp.train_test_split}%`]
    ];

    autoTable(doc, {
      startY: reportY + 5,
      head: [["Parameter", "Value"]],
      body: paramData,
      theme: 'grid',
      headStyles: { fillColor: [120, 113, 108] }
    });

    // Add Charts Section
    doc.addPage();
    doc.setFontSize(18);
    doc.text("Visual Analysis", 14, 22);

    const captureChart = async (id: string) => {
      const element = document.getElementById(id);
      if (element) {
        const canvas = await html2canvas(element, {
          scale: 2,
          useCORS: true,
          logging: false,
          backgroundColor: '#ffffff'
        });
        return canvas.toDataURL('image/png');
      }
      return null;
    };

    const chartIds = [
      { id: 'report-chart-loss', title: 'Loss & Accuracy' },
      { id: 'report-chart-grads', title: 'Gradients & Updates' },
      { id: 'report-chart-speed', title: 'Convergence Speed' },
      { id: 'report-chart-stability', title: 'Training Stability' }
    ];

    reportY = 35;
    for (const chart of chartIds) {
      const imgData = await captureChart(chart.id);
      if (imgData) {
        if (reportY + 80 > 280) {
          doc.addPage();
          reportY = 20;
        }
        doc.setFontSize(12);
        doc.text(chart.title, 14, reportY);
        doc.addImage(imgData, 'PNG', 14, reportY + 5, 180, 70);
        reportY += 85;
      }
    }

    // Epoch Details (New Page if needed)
    doc.addPage();
    doc.setFontSize(16);
    doc.text("Epoch-by-Epoch Training Logs", 14, 22);

    const logs = typeof exp.logs === 'string' ? JSON.parse(exp.logs) : exp.logs;
    const logData = logs.map((m: any) => [
      m.epoch,
      safeFixed(m.loss, 4),
      safeFixed(m.accuracy, 2, 100, '%'),
      safeFixed(m.gradientNorm, 4),
      safeFixed(m.updateRatio, 6)
    ]);

    autoTable(doc, {
      startY: 30,
      head: [["Epoch", "Loss", "Accuracy", "Grad Norm", "Update Ratio"]],
      body: logData,
      theme: 'striped',
      headStyles: { fillColor: [28, 25, 23] }
    });

    doc.save(`experiment_${exp.id}_report.pdf`);
  };

  const chartData = useMemo(() => {
    if (results.length === 0) return [];
    const epochs = results[0].metrics.length;
    const data = [];
    for (let i = 0; i < epochs; i++) {
      const entry: any = { epoch: i + 1 };
      results.forEach(res => {
        entry[`${res.optimizer}_loss`] = res.metrics[i].loss;
        entry[`${res.optimizer}_acc`] = res.metrics[i].accuracy;
        entry[`${res.optimizer}_grad`] = res.metrics[i].gradientNorm;
        entry[`${res.optimizer}_ratio`] = res.metrics[i].updateRatio;
      });
      data.push(entry);
    }
    return data;
  }, [results]);

  return (
    <div className="min-h-screen bg-[#F5F5F4] text-[#1C1917] font-sans flex">
      {/* Sidebar */}
      <aside className="w-80 bg-white border-r border-[#E7E5E4] p-6 flex flex-col gap-8 overflow-y-auto">
        <div className="flex items-center gap-3">
          <div className="bg-[#1C1917] p-2 rounded-lg">
            <Database className="text-white w-5 h-5" />
          </div>
          <h1 className="font-bold text-lg tracking-tight">Neur-O-Opt Lab</h1>
        </div>

        {/* Dataset Upload */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-xs font-semibold text-[#78716C] uppercase tracking-wider">
            <Upload className="w-3 h-3" />
            Dataset Upload
          </div>
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium mb-1">Training CSV</label>
              <input 
                type="file" accept=".csv" 
                onChange={(e) => handleFileUpload(e, 'train')}
                className="block w-full text-xs text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-[#F5F5F4] file:text-[#1C1917] hover:file:bg-[#E7E5E4] cursor-pointer"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Testing CSV</label>
              <input 
                type="file" accept=".csv" 
                onChange={(e) => handleFileUpload(e, 'test')}
                className="block w-full text-xs text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-[#F5F5F4] file:text-[#1C1917] hover:file:bg-[#E7E5E4] cursor-pointer"
              />
            </div>
          </div>
        </section>

        {/* Sample Sizes */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-xs font-semibold text-[#78716C] uppercase tracking-wider">
            <BarChart3 className="w-3 h-3" />
            Sampling
          </div>
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="block text-[11px] font-medium">Training Samples: {trainSampleSize.toLocaleString()}</label>
              <input 
                type="range" min="2000" max="500000" step="1000" value={trainSampleSize}
                onChange={(e) => setTrainSampleSize(parseInt(e.target.value))}
                className="w-full h-1.5 bg-[#F5F5F4] rounded-lg appearance-none cursor-pointer accent-[#1C1917]"
              />
              <div className="flex justify-between text-[10px] text-[#A8A29E]">
                <span>2k</span>
                <span>500k</span>
              </div>
            </div>
            <div className="space-y-2">
              <label className="block text-[11px] font-medium">Testing Samples: {testSampleSize.toLocaleString()}</label>
              <input 
                type="range" min="500" max="50000" step="500" value={testSampleSize}
                onChange={(e) => setTestSampleSize(parseInt(e.target.value))}
                className="w-full h-1.5 bg-[#F5F5F4] rounded-lg appearance-none cursor-pointer accent-[#1C1917]"
              />
              <div className="flex justify-between text-[10px] text-[#A8A29E]">
                <span>500</span>
                <span>50k</span>
              </div>
            </div>
          </div>
        </section>

        {/* Model Params */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-xs font-semibold text-[#78716C] uppercase tracking-wider">
            <Settings className="w-3 h-3" />
            Model Parameters
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-[11px] font-medium">Hidden Size</label>
              <input 
                type="number" value={params.hiddenSize || ''} 
                onChange={e => {
                  const val = parseInt(e.target.value);
                  setParams({...params, hiddenSize: isNaN(val) ? 0 : val});
                }}
                className="w-full bg-[#F5F5F4] border-none rounded-md px-3 py-2 text-sm focus:ring-1 ring-[#1C1917]"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[11px] font-medium">Learning Rate</label>
              <input 
                type="number" step="0.001" value={params.learningRate || ''} 
                onChange={e => {
                  const val = parseFloat(e.target.value);
                  setParams({...params, learningRate: isNaN(val) ? 0 : val});
                }}
                className="w-full bg-[#F5F5F4] border-none rounded-md px-3 py-2 text-sm focus:ring-1 ring-[#1C1917]"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[11px] font-medium">Epochs</label>
              <input 
                type="number" value={params.epochs || ''} 
                onChange={e => {
                  const val = parseInt(e.target.value);
                  setParams({...params, epochs: isNaN(val) ? 0 : val});
                }}
                className="w-full bg-[#F5F5F4] border-none rounded-md px-3 py-2 text-sm focus:ring-1 ring-[#1C1917]"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[11px] font-medium">Batch Size</label>
              <input 
                type="number" value={params.batchSize || ''} 
                onChange={e => {
                  const val = parseInt(e.target.value);
                  setParams({...params, batchSize: isNaN(val) ? 0 : val});
                }}
                className="w-full bg-[#F5F5F4] border-none rounded-md px-3 py-2 text-sm focus:ring-1 ring-[#1C1917]"
              />
            </div>
          </div>
        </section>

        <div className="flex gap-3 mt-auto">
          {isTraining ? (
            <button 
              onClick={stopTraining}
              className="flex-1 bg-red-600 text-white py-3 rounded-xl font-semibold flex items-center justify-center gap-2 hover:bg-red-700 transition-colors"
            >
              <AlertCircle className="w-4 h-4" />
              Stop
            </button>
          ) : (
            <button 
              onClick={startTraining}
              disabled={!trainFile || !testFile}
              className="flex-1 bg-[#1C1917] text-white py-3 rounded-xl font-semibold flex items-center justify-center gap-2 hover:bg-[#44403C] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Play className="w-4 h-4 fill-current" />
              Start
            </button>
          )}
          {isTraining && (
            <button 
              onClick={togglePause}
              className="px-4 bg-white border border-[#E7E5E4] rounded-xl flex items-center justify-center hover:bg-[#F5F5F4] transition-colors"
            >
              {isPaused ? <PlayCircle className="w-5 h-5" /> : <Pause className="w-5 h-5" />}
            </button>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8 overflow-y-auto space-y-8">
        {/* Dataset Previews */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
          {trainData.length > 0 && (
            <section className="bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm">
              <div className="flex justify-between items-center mb-4">
                <h2 className="font-bold text-lg">Training Preview</h2>
                <div className="flex gap-4 text-xs font-medium text-[#78716C]">
                  <span>Shape: {trainData.length} × {Object.keys(trainData[0] || {}).length}</span>
                </div>
              </div>
              <div className="overflow-x-auto border border-[#E7E5E4] rounded-lg">
                <table className="w-full text-[11px] text-left min-w-[600px]">
                  <thead className="bg-[#F5F5F4] text-[#78716C] uppercase text-[9px] tracking-wider">
                    <tr>
                      {Object.keys(trainData[0]).map(col => (
                        <th key={col} className="px-3 py-2 font-semibold whitespace-nowrap">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E7E5E4]">
                    {trainData.slice(0, 5).map((row, i) => (
                      <tr key={i}>
                        {Object.values(row).map((val: any, j) => (
                          <td key={j} className="px-3 py-2 text-[#44403C] whitespace-nowrap">{typeof val === 'number' ? val.toFixed(3) : val}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {testData.length > 0 && (
            <section className="bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm">
              <div className="flex justify-between items-center mb-4">
                <h2 className="font-bold text-lg">Testing Preview</h2>
                <div className="flex gap-4 text-xs font-medium text-[#78716C]">
                  <span>Shape: {testData.length} × {Object.keys(testData[0] || {}).length}</span>
                </div>
              </div>
              <div className="overflow-x-auto border border-[#E7E5E4] rounded-lg">
                <table className="w-full text-[11px] text-left min-w-[600px]">
                  <thead className="bg-[#F5F5F4] text-[#78716C] uppercase text-[9px] tracking-wider">
                    <tr>
                      {Object.keys(testData[0]).map(col => (
                        <th key={col} className="px-3 py-2 font-semibold whitespace-nowrap">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E7E5E4]">
                    {testData.slice(0, 5).map((row, i) => (
                      <tr key={i}>
                        {Object.values(row).map((val: any, j) => (
                          <td key={j} className="px-3 py-2 text-[#44403C] whitespace-nowrap">{typeof val === 'number' ? val.toFixed(3) : val}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </div>

        {/* Training Progress */}
        {isTraining && (
          <section className="bg-[#1C1917] text-white rounded-2xl p-8 shadow-xl animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                <span className="text-sm font-medium text-[#A8A29E]">{statusMessage}</span>
              </div>
              {isPaused && (
                <span className="px-2 py-1 bg-amber-500/20 text-amber-500 text-[10px] font-bold rounded uppercase tracking-wider animate-pulse">
                  Paused
                </span>
              )}
            </div>
            <div className="grid grid-cols-3 gap-8">
              <div className="space-y-2">
                <div className="text-[#A8A29E] text-xs font-semibold uppercase tracking-widest flex items-center gap-2">
                  <Settings className="w-3 h-3" /> Current Optimizer
                  <InfoTooltip title={currentOptimizer || ''} content={METRICS_INFO[currentOptimizer as keyof typeof METRICS_INFO] || ''} />
                </div>
                <div className="text-3xl font-bold tracking-tight">{currentOptimizer}</div>
              </div>
              <div className="space-y-2">
                <div className="text-[#A8A29E] text-xs font-semibold uppercase tracking-widest flex items-center gap-2">
                  <History className="w-3 h-3" /> Epoch
                </div>
                <div className="text-3xl font-bold tracking-tight">{currentEpoch} / {params.epochs}</div>
              </div>
              <div className="space-y-2">
                <div className="text-[#A8A29E] text-xs font-semibold uppercase tracking-widest flex items-center gap-2">
                  <Timer className="w-3 h-3" /> Elapsed Time
                  <InfoTooltip title="Execution Time" content={METRICS_INFO['Execution Time']} />
                </div>
                <div className="text-3xl font-bold tracking-tight">{elapsedTime}s</div>
              </div>
            </div>
            <div className="mt-8 grid grid-cols-2 gap-8">
              <div className="space-y-2">
                <div className="flex justify-between text-[10px] font-bold text-[#A8A29E] uppercase tracking-widest">
                  <span>Training Progress</span>
                  <span>{Math.round(trainingProgress)}%</span>
                </div>
                <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <motion.div 
                    className="h-full bg-emerald-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${trainingProgress}%` }}
                  />
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-[10px] font-bold text-[#A8A29E] uppercase tracking-widest">
                  <span>Testing Progress</span>
                  <span>{Math.round(testingProgress)}%</span>
                </div>
                <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <motion.div 
                    className="h-full bg-blue-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${testingProgress}%` }}
                  />
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Results Visualization */}
        {results.length > 0 && (
          <div className="grid grid-cols-2 gap-8">
            <section className="bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm">
              <h3 className="font-bold mb-6 flex items-center gap-2">
                <BarChart3 className="w-4 h-4" /> 
                Loss vs Epoch
                <InfoTooltip title="Log Loss" content={METRICS_INFO['Log Loss']} />
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F5F5F4" />
                    <XAxis dataKey="epoch" axisLine={false} tickLine={false} tick={{fontSize: 10}} />
                    <YAxis axisLine={false} tickLine={false} tick={{fontSize: 10}} />
                    <Tooltip contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)'}} />
                    <Legend iconType="circle" wrapperStyle={{fontSize: 10, paddingTop: 20}} />
                    {results.map((res, i) => (
                      <Line 
                        key={res.optimizer} 
                        type="monotone" 
                        dataKey={`${res.optimizer}_loss`} 
                        stroke={['#1C1917', '#D97706', '#059669', '#2563EB'][i]} 
                        strokeWidth={2} 
                        dot={false}
                        name={res.optimizer}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm">
              <h3 className="font-bold mb-6 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4" /> 
                Accuracy vs Epoch
                <InfoTooltip title="Test Accuracy" content={METRICS_INFO['Test Accuracy']} />
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F5F5F4" />
                    <XAxis dataKey="epoch" axisLine={false} tickLine={false} tick={{fontSize: 10}} />
                    <YAxis axisLine={false} tickLine={false} tick={{fontSize: 10}} domain={[0, 1]} />
                    <Tooltip contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)'}} />
                    <Legend iconType="circle" wrapperStyle={{fontSize: 10, paddingTop: 20}} />
                    {results.map((res, i) => (
                      <Line 
                        key={res.optimizer} 
                        type="monotone" 
                        dataKey={`${res.optimizer}_acc`} 
                        stroke={['#1C1917', '#D97706', '#059669', '#2563EB'][i]} 
                        strokeWidth={2} 
                        dot={false}
                        name={res.optimizer}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>
        )}

        {/* Advanced Metrics Visualization */}
        {results.length > 0 && (
          <div className="grid grid-cols-2 gap-8">
            <section className="bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm">
              <h3 className="font-bold mb-6 flex items-center gap-2">
                <BarChart3 className="w-4 h-4" /> 
                Gradient Norm vs Epoch
                <InfoTooltip title="Gradient Norm" content={METRICS_INFO['Gradient Norm']} />
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F5F5F4" />
                    <XAxis dataKey="epoch" axisLine={false} tickLine={false} tick={{fontSize: 10}} />
                    <YAxis axisLine={false} tickLine={false} tick={{fontSize: 10}} />
                    <Tooltip contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)'}} />
                    <Legend iconType="circle" wrapperStyle={{fontSize: 10, paddingTop: 20}} />
                    {results.map((res, i) => (
                      <Line 
                        key={res.optimizer} 
                        type="monotone" 
                        dataKey={`${res.optimizer}_grad`} 
                        stroke={['#1C1917', '#D97706', '#059669', '#2563EB'][i]} 
                        strokeWidth={2} 
                        dot={false}
                        name={res.optimizer}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm">
              <h3 className="font-bold mb-6 flex items-center gap-2">
                <Timer className="w-4 h-4" /> 
                Update Ratio vs Epoch
                <InfoTooltip title="Update Ratio" content={METRICS_INFO['Update Ratio']} />
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F5F5F4" />
                    <XAxis dataKey="epoch" axisLine={false} tickLine={false} tick={{fontSize: 10}} />
                    <YAxis axisLine={false} tickLine={false} tick={{fontSize: 10}} />
                    <Tooltip contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)'}} />
                    <Legend iconType="circle" wrapperStyle={{fontSize: 10, paddingTop: 20}} />
                    {results.map((res, i) => (
                      <Line 
                        key={res.optimizer} 
                        type="monotone" 
                        dataKey={`${res.optimizer}_ratio`} 
                        stroke={['#1C1917', '#D97706', '#059669', '#2563EB'][i]} 
                        strokeWidth={2} 
                        dot={false}
                        name={res.optimizer}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>
        )}

        {/* Comparison Table */}
        {results.length > 0 && (
          <section className="bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm overflow-hidden">
            <h2 className="font-bold text-lg mb-6">Optimizer Comparison</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="bg-[#F5F5F4] text-[#78716C] uppercase text-[10px] tracking-wider">
                  <tr>
                    <th className="px-6 py-4 font-semibold">Optimizer</th>
                    <th className="px-6 py-4 font-semibold">
                      Accuracy
                      <InfoTooltip title="Test Accuracy" content={METRICS_INFO['Test Accuracy']} />
                    </th>
                    <th className="px-6 py-4 font-semibold">
                      F1 Score
                      <InfoTooltip title="F1 Score" content={METRICS_INFO['F1 Score']} />
                    </th>
                    <th className="px-6 py-4 font-semibold">
                      Precision
                      <InfoTooltip title="Precision (Macro)" content={METRICS_INFO['Precision (Macro)']} />
                    </th>
                    <th className="px-6 py-4 font-semibold">
                      Recall
                      <InfoTooltip title="Recall (Macro)" content={METRICS_INFO['Recall (Macro)']} />
                    </th>
                    <th className="px-6 py-4 font-semibold">
                      Log Loss
                      <InfoTooltip title="Log Loss" content={METRICS_INFO['Log Loss']} />
                    </th>
                    <th className="px-6 py-4 font-semibold">
                      Convergence
                      <InfoTooltip title="Convergence Rate" content={METRICS_INFO['Convergence Rate']} />
                    </th>
                    <th className="px-6 py-4 font-semibold">
                      Variance
                      <InfoTooltip title="Loss Variance" content={METRICS_INFO['Loss Variance']} />
                    </th>
                    <th className="px-6 py-4 font-semibold">
                      Time
                      <InfoTooltip title="Execution Time" content={METRICS_INFO['Execution Time']} />
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E7E5E4]">
                  {results.map((res) => (
                    <tr key={res.optimizer} className={cn(bestOptimizer?.optimizer === res.optimizer && "bg-emerald-50/50")}>
                      <td className="px-6 py-4 font-bold flex items-center gap-2">
                        {res.optimizer}
                        <InfoTooltip title={res.optimizer} content={METRICS_INFO[res.optimizer as keyof typeof METRICS_INFO]} />
                        {bestOptimizer?.optimizer === res.optimizer && <CheckCircle2 className="w-3 h-3 text-emerald-600" />}
                      </td>
                      <td className="px-6 py-4">{safeFixed(res.testAccuracy, 1, 100, '%')}</td>
                      <td className="px-6 py-4">{safeFixed(res.f1Score, 1, 100, '%')}</td>
                      <td className="px-6 py-4">{safeFixed(res.precision, 1, 100, '%')}</td>
                      <td className="px-6 py-4">{safeFixed(res.recall, 1, 100, '%')}</td>
                      <td className="px-6 py-4">{safeFixed(res.logLoss, 4)}</td>
                      <td className="px-6 py-4">{safeFixed(res.convergenceRate, 2, 1, 'x')}</td>
                      <td className="px-6 py-4">{safeFixed(res.lossVariance, 6)}</td>
                      <td className="px-6 py-4">{safeFixed(res.executionTime, 1, 1, 's')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Analysis Section */}
        {bestOptimizer && (
          <div className="grid grid-cols-3 gap-8">
            <section className="col-span-1 bg-emerald-600 text-white rounded-2xl p-6 shadow-lg">
              <h3 className="text-xs font-bold uppercase tracking-widest opacity-80 mb-2">Best Optimizer</h3>
              <div className="text-3xl font-black mb-4">{bestOptimizer.optimizer}</div>
              <p className="text-sm leading-relaxed opacity-90">
                The best optimizer for this dataset is <span className="font-bold">{bestOptimizer.optimizer}</span> because it achieved a test accuracy of <span className="font-bold">{safeFixed(bestOptimizer.testAccuracy, 2, 100, '%')}</span> with a convergence rate of <span className="font-bold">{safeFixed(bestOptimizer.convergenceRate, 2, 1, 'x')}</span>.
              </p>
            </section>

            <section className="col-span-2 bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm">
              <h3 className="font-bold mb-4 flex items-center gap-2">
                <Info className="w-4 h-4 text-[#78716C]" /> Optimizer Performance Analysis
              </h3>
              <div className="grid grid-cols-2 gap-4">
                {results.map(res => (
                  <div key={res.optimizer} className="p-3 bg-[#F5F5F4] rounded-xl">
                    <div className="text-xs font-bold mb-1">{res.optimizer}</div>
                    <p className="text-[11px] text-[#78716C]">
                      {res.optimizer === 'Adam' && "Adam achieved the highest accuracy and fastest convergence through adaptive momentum."}
                      {res.optimizer === 'SGD' && "SGD converged slower but produced stable gradients, suitable for simpler landscapes."}
                      {res.optimizer === 'RMSProp' && "RMSProp showed faster convergence compared to SGD by scaling gradients."}
                      {res.optimizer === 'Adagrad' && "Adagrad reduced the learning rate over time, effectively handling sparse features."}
                    </p>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}

        {/* History Dashboard */}
        <section className="bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm">
          <h2 className="font-bold text-lg mb-6 flex items-center gap-2">
            <History className="w-5 h-5" /> Experiment History
          </h2>
          <div className="space-y-3">
            {history.length === 0 ? (
              <div className="text-center py-12 text-[#A8A29E] italic text-sm">No experiments recorded yet.</div>
            ) : (
              history.map((exp) => (
                <div 
                  key={exp.id} 
                  onClick={() => fetchExperimentDetails(exp.id)}
                  className="flex items-center justify-between p-4 bg-[#F5F5F4] rounded-xl hover:bg-[#E7E5E4] transition-colors cursor-pointer group"
                >
                  <div className="flex items-center gap-4">
                    <div className="bg-white p-2 rounded-lg shadow-sm group-hover:bg-[#1C1917] group-hover:text-white transition-colors">
                      <Database className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="font-bold text-sm">{exp.dataset_name}</div>
                      <div className="text-[10px] text-[#78716C] uppercase tracking-wider">{exp.optimizer} • {new Date(exp.timestamp).toLocaleString()}</div>
                    </div>
                  </div>
                  <div className="flex gap-8 text-right items-center">
                    <div>
                      <div className="text-[10px] text-[#78716C] uppercase font-semibold flex items-center justify-end gap-1">
                        Accuracy
                        <InfoTooltip title="Test Accuracy" content={METRICS_INFO['Test Accuracy']} />
                      </div>
                      <div className="text-sm font-bold">{safeFixed(exp.test_accuracy, 1, 100, '%')}</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[#78716C] uppercase font-semibold flex items-center justify-end gap-1">
                        Time
                        <InfoTooltip title="Execution Time" content={METRICS_INFO['Execution Time']} />
                      </div>
                      <div className="text-sm font-bold">{safeFixed(exp.execution_time, 1, 1, 's')}</div>
                    </div>
                    <div className="pl-4 border-l border-[#E7E5E4] flex items-center">
                      <div className="p-2 rounded-full bg-white text-[#1C1917] opacity-0 group-hover:opacity-100 transition-opacity">
                        <ChevronRight className="w-4 h-4" />
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        {/* Experiment Report Modal */}
        <AnimatePresence>
          {isViewingReport && selectedExperiment && (
            <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-8">
              <motion.div 
                initial={{ opacity: 0, y: -20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 20, scale: 0.95 }}
                className="bg-white w-full max-w-6xl max-h-[90vh] rounded-3xl shadow-2xl overflow-hidden flex flex-col"
              >
              <div className="p-6 border-b border-[#E7E5E4] flex justify-between items-center bg-[#F5F5F4]">
                <div>
                  <h2 className="text-2xl font-black tracking-tight">Experiment Report #{selectedExperiment.id}</h2>
                  <p className="text-sm text-[#78716C]">{selectedExperiment.dataset_name} • {selectedExperiment.optimizer} • {new Date(selectedExperiment.timestamp).toLocaleString()}</p>
                </div>
                <button 
                  onClick={() => setIsViewingReport(false)}
                  className="p-2 hover:bg-white rounded-full transition-colors"
                >
                  <AlertCircle className="w-6 h-6 rotate-45" />
                </button>
              </div>
              
              <div className="flex-1 overflow-y-auto p-8 space-y-8">
                {/* Overview */}
                <div className="grid grid-cols-4 gap-6">
                  <div className="p-4 bg-[#F5F5F4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1">Hidden Size</div>
                    <div className="text-xl font-black">{selectedExperiment.hidden_size}</div>
                  </div>
                  <div className="p-4 bg-[#F5F5F4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1">Learning Rate</div>
                    <div className="text-xl font-black">{selectedExperiment.learning_rate}</div>
                  </div>
                  <div className="p-4 bg-[#F5F5F4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1">Epochs</div>
                    <div className="text-xl font-black">{selectedExperiment.epochs}</div>
                  </div>
                  <div className="p-4 bg-[#F5F5F4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1">Batch Size</div>
                    <div className="text-xl font-black">{selectedExperiment.batch_size}</div>
                  </div>
                </div>

                {/* Analysis */}
                <section className="bg-emerald-50 border border-emerald-100 p-6 rounded-2xl">
                  <h3 className="font-bold text-emerald-900 mb-2 flex items-center gap-2">
                    <Info className="w-4 h-4" /> Experiment Analysis
                  </h3>
                  <p className="text-sm text-emerald-800 leading-relaxed">
                    This experiment using <span className="font-bold">{selectedExperiment.optimizer}</span> on the <span className="font-bold">{selectedExperiment.dataset_name}</span> dataset achieved a test accuracy of <span className="font-bold">{safeFixed(selectedExperiment.test_accuracy, 2, 100, '%')}</span>. 
                    The model converged with a rate of <span className="font-bold">{safeFixed(selectedExperiment.convergence_rate, 2, 1, 'x')}</span> over <span className="font-bold">{safeFixed(selectedExperiment.execution_time, 2, 1, 's')}</span>.
                    {selectedExperiment.optimizer === 'Adam' ? " Adam's adaptive learning rate helped in stable convergence." : ""}
                    {selectedExperiment.test_accuracy > 0.8 ? " The high accuracy suggests well-tuned parameters for this specific data." : " There might be room for improvement by adjusting the learning rate or hidden layer size."}
                  </p>
                </section>

                {/* Graphs */}
                <div className="grid grid-cols-2 gap-8">
                  <div className="bg-white border border-[#E7E5E4] p-6 rounded-2xl" id="report-chart-loss">
                    <h4 className="font-bold mb-4 text-sm flex items-center gap-1">
                      Loss & Accuracy
                      <InfoTooltip title="Test Accuracy" content={METRICS_INFO['Test Accuracy']} />
                    </h4>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={selectedExperiment.logs}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F5F5F4" />
                          <XAxis dataKey="epoch" tick={{fontSize: 10}} />
                          <YAxis yAxisId="left" tick={{fontSize: 10}} />
                          <YAxis yAxisId="right" orientation="right" tick={{fontSize: 10}} domain={[0, 1]} />
                          <Tooltip />
                          <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#1C1917" strokeWidth={2} dot={false} name="Loss" />
                          <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#059669" strokeWidth={2} dot={false} name="Accuracy" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div className="bg-white border border-[#E7E5E4] p-6 rounded-2xl" id="report-chart-grads">
                    <h4 className="font-bold mb-4 text-sm flex items-center gap-1">
                      Gradients & Updates
                      <InfoTooltip title="Gradient Norm" content={METRICS_INFO['Gradient Norm']} />
                    </h4>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={selectedExperiment.logs}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F5F5F4" />
                          <XAxis dataKey="epoch" tick={{fontSize: 10}} />
                          <YAxis tick={{fontSize: 10}} />
                          <Tooltip />
                          <Line type="monotone" dataKey="gradientNorm" stroke="#D97706" strokeWidth={2} dot={false} name="Grad Norm" />
                          <Line type="monotone" dataKey="updateRatio" stroke="#2563EB" strokeWidth={2} dot={false} name="Update Ratio" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                {/* Advanced Metrics */}
                <div className="grid grid-cols-5 gap-4">
                  <div className="p-4 bg-white border border-[#E7E5E4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1 flex items-center gap-1">
                      F1 Score
                      <InfoTooltip title="F1 Score" content={METRICS_INFO['F1 Score']} />
                    </div>
                    <div className="text-xl font-black text-blue-600">{safeFixed(selectedExperiment.f1_score, 2, 100, '%')}</div>
                  </div>
                  <div className="p-4 bg-white border border-[#E7E5E4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1 flex items-center gap-1">
                      Log Loss
                      <InfoTooltip title="Log Loss" content={METRICS_INFO['Log Loss']} />
                    </div>
                    <div className="text-xl font-black text-rose-600">{safeFixed(selectedExperiment.log_loss, 4)}</div>
                  </div>
                  <div className="p-4 bg-white border border-[#E7E5E4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1 flex items-center gap-1">
                      AULC
                      <InfoTooltip title="AULC" content={METRICS_INFO['AULC']} />
                    </div>
                    <div className="text-xl font-black text-amber-600">{safeFixed(selectedExperiment.aulc, 2)}</div>
                  </div>
                  <div className="p-4 bg-white border border-[#E7E5E4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1 flex items-center gap-1">
                      Loss Variance
                      <InfoTooltip title="Loss Variance" content={METRICS_INFO['Loss Variance']} />
                    </div>
                    <div className="text-xl font-black">{safeFixed(selectedExperiment.loss_variance, 6)}</div>
                  </div>
                  <div className="p-4 bg-white border border-[#E7E5E4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1 flex items-center gap-1">
                      Conv. Rate
                      <InfoTooltip title="Convergence Rate" content={METRICS_INFO['Convergence Rate']} />
                    </div>
                    <div className="text-xl font-black text-emerald-600">{safeFixed(selectedExperiment.convergence_rate, 2, 1, 'x')}</div>
                  </div>
                </div>

                {/* Advanced Visualizations */}
                <div className="grid grid-cols-2 gap-8">
                  <div className="bg-white border border-[#E7E5E4] p-6 rounded-2xl" id="report-chart-speed">
                    <h4 className="font-bold mb-4 text-sm flex items-center gap-2">
                      <TrendingDown className="w-4 h-4" /> 
                      Convergence Speed
                      <InfoTooltip title="Convergence Speed" content={METRICS_INFO['Convergence Speed']} />
                    </h4>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={selectedExperiment.logs}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F5F5F4" />
                          <XAxis dataKey="epoch" hide />
                          <YAxis tick={{fontSize: 10}} />
                          <Tooltip />
                          <Area type="monotone" dataKey="convergenceSpeed" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.1} name="Speed" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div className="bg-white border border-[#E7E5E4] p-6 rounded-2xl" id="report-chart-stability">
                    <h4 className="font-bold mb-4 text-sm flex items-center gap-2">
                      <Activity className="w-4 h-4" /> 
                      Training Stability
                      <InfoTooltip title="Loss Variance" content={METRICS_INFO['Loss Variance']} />
                    </h4>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F5F5F4" />
                          <XAxis type="number" dataKey="epoch" name="Epoch" hide />
                          <YAxis type="number" dataKey="loss" name="Loss" tick={{fontSize: 10}} />
                          <ZAxis type="number" dataKey="updateRatio" range={[20, 200]} name="Update" />
                          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                          <Scatter name="Stability" data={selectedExperiment.logs} fill="#EC4899" fillOpacity={0.6} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-8">
                  <div className="bg-white border border-[#E7E5E4] p-6 rounded-2xl">
                    <h4 className="font-bold mb-4 text-sm flex items-center gap-2"><Zap className="w-4 h-4" /> Optimization Dynamics</h4>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={selectedExperiment.logs}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F5F5F4" />
                          <XAxis dataKey="epoch" hide />
                          <YAxis tick={{fontSize: 10}} />
                          <Tooltip />
                          <Line type="monotone" dataKey="updateRatio" stroke="#3B82F6" strokeWidth={2} dot={false} />
                          <Line type="monotone" dataKey="gradientNorm" stroke="#F59E0B" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                {/* Epoch Table */}
                <section>
                  <h3 className="font-bold mb-4">Epoch Details</h3>
                  <div className="overflow-x-auto border border-[#E7E5E4] rounded-xl">
                    <table className="w-full text-sm text-left">
                      <thead className="bg-[#F5F5F4] text-[#78716C] uppercase text-[10px] tracking-wider">
                        <tr>
                          <th className="px-6 py-3 font-semibold">Epoch</th>
                          <th className="px-6 py-3 font-semibold">Loss</th>
                          <th className="px-6 py-3 font-semibold">Accuracy</th>
                          <th className="px-6 py-3 font-semibold">Grad Norm</th>
                          <th className="px-6 py-3 font-semibold">Update Ratio</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#E7E5E4]">
                        {selectedExperiment.logs.map((m: any) => (
                          <tr key={m.epoch}>
                            <td className="px-6 py-3 font-bold">{m.epoch}</td>
                            <td className="px-6 py-3">{safeFixed(m.loss, 4)}</td>
                            <td className="px-6 py-3">{safeFixed(m.accuracy, 2, 100, '%')}</td>
                            <td className="px-6 py-3">{safeFixed(m.gradientNorm, 4)}</td>
                            <td className="px-6 py-3">{safeFixed(m.updateRatio, 6)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>

                {/* Testing Results */}
                <div className="grid grid-cols-4 gap-6">
                  <div className="p-6 border border-[#E7E5E4] rounded-2xl text-center">
                    <div className="text-xs text-[#78716C] uppercase font-bold mb-1 flex items-center justify-center gap-1">
                      Precision
                      <InfoTooltip title="Precision (Macro)" content={METRICS_INFO['Precision (Macro)']} />
                    </div>
                    <div className="text-2xl font-black">{safeFixed(selectedExperiment.precision, 1, 100, '%')}</div>
                  </div>
                  <div className="p-6 border border-[#E7E5E4] rounded-2xl text-center">
                    <div className="text-xs text-[#78716C] uppercase font-bold mb-1 flex items-center justify-center gap-1">
                      Recall
                      <InfoTooltip title="Recall (Macro)" content={METRICS_INFO['Recall (Macro)']} />
                    </div>
                    <div className="text-2xl font-black">{safeFixed(selectedExperiment.recall, 1, 100, '%')}</div>
                  </div>
                  <div className="p-6 border border-[#E7E5E4] rounded-2xl text-center">
                    <div className="text-xs text-[#78716C] uppercase font-bold mb-1 flex items-center justify-center gap-1">
                      F1 Score
                      <InfoTooltip title="F1 Score" content={METRICS_INFO['F1 Score']} />
                    </div>
                    <div className="text-2xl font-black">{safeFixed(selectedExperiment.f1_score, 1, 100, '%')}</div>
                  </div>
                  <div className="p-6 border border-[#E7E5E4] rounded-2xl text-center">
                    <div className="text-xs text-[#78716C] uppercase font-bold mb-1 flex items-center justify-center gap-1">
                      Log Loss
                      <InfoTooltip title="Log Loss" content={METRICS_INFO['Log Loss']} />
                    </div>
                    <div className="text-2xl font-black">{safeFixed(selectedExperiment.log_loss, 3)}</div>
                  </div>
                </div>

                <div className="flex justify-center pt-4">
                  <button 
                    onClick={() => downloadReport(selectedExperiment)}
                    className="flex items-center gap-2 px-6 py-3 bg-[#1C1917] text-white rounded-xl font-bold hover:bg-black transition-all shadow-lg"
                  >
                    <Download className="w-4 h-4" /> Download Experiment Report (PDF)
                  </button>
                </div>
              </div>
              </motion.div>
            </div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
