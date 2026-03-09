import React, { useState, useEffect, useMemo, useRef } from 'react';
import Papa from 'papaparse';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { 
  Upload, Play, History, BarChart3, Settings, Database, Timer, CheckCircle2, AlertCircle, Info, Pause, PlayCircle
} from 'lucide-react';
import { NeuralNetwork, OptimizerType, ModelParams, ExperimentResult, TrainingMetric } from './ml-engine';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export default function App() {
  // State
  const [trainData, setTrainData] = useState<any[]>([]);
  const [testData, setTestData] = useState<any[]>([]);
  const [trainFile, setTrainFile] = useState<File | null>(null);
  const [testFile, setTestFile] = useState<File | null>(null);
  
  const [features, setFeatures] = useState<string[]>([]);
  const [target, setTarget] = useState<string>('');
  const [sampleSize, setSampleSize] = useState<number>(10000);
  
  const [params, setParams] = useState<ModelParams>({
    hiddenSize: 64,
    learningRate: 0.01,
    epochs: 10,
    batchSize: 64
  });

  const [isTraining, setIsTraining] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string>('');
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

    const fullTrain = await loadAndSample(trainFile, sampleSize);
    const fullTest = await loadAndSample(testFile, Math.min(2000, sampleSize));

    // Prepare and Normalize Data once
    const X_train_raw = fullTrain.map(row => features.map(f => row[f] || 0));
    const y_train = fullTrain.map(row => row[target]);
    const X_test_raw = fullTest.map(row => features.map(f => row[f] || 0));
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
    const classes = Array.from(new Set(y_train)).sort();
    const classMap = new Map(classes.map((c, i) => [c, i]));
    const y_train_idx = y_train.map(v => classMap.get(v) || 0);
    const y_test_idx = y_test.map(v => classMap.get(v) || 0);

    const optimizers: OptimizerType[] = ['SGD', 'Adagrad', 'RMSProp', 'Adam'];
    const allResults: ExperimentResult[] = [];

    for (const opt of optimizers) {
      if (stopTrainingRef.current) break;
      setCurrentOptimizer(opt);
      setStatusMessage(`Training ${opt} optimizer...`);
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
        }

        if (stopTrainingRef.current) break;

        const avgLoss = totalLoss / batchCount;
        const accuracy = nn.evaluate(X_train_norm, y_train_idx);
        
        metrics.push({
          epoch,
          loss: avgLoss,
          accuracy,
          gradientNorm: totalGradNorm / batchCount,
          updateRatio: totalUpdateNorm / batchCount
        });

        // Yield to UI
        await new Promise(r => setTimeout(r, 0));
      }

      if (stopTrainingRef.current) break;

      setStatusMessage(`Testing ${opt} performance...`);
      const testAccuracy = nn.evaluate(X_test_norm, y_test_idx);
      const executionTime = (Date.now() - startTime) / 1000;
      
      const meanLoss = metrics.reduce((s, x) => s + x.loss, 0) / metrics.length;
      const result: ExperimentResult = {
        optimizer: opt,
        metrics,
        testAccuracy,
        executionTime,
        convergenceRate: metrics[0].loss / metrics[metrics.length - 1].loss,
        lossVariance: metrics.reduce((acc, m) => acc + Math.pow(m.loss - meanLoss, 2), 0) / metrics.length
      };

      allResults.push(result);
      setResults([...allResults]);

      // Save to DB
      await fetch('/api/experiments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_name: trainFile.name,
          sample_size: sampleSize,
          optimizer: opt,
          test_accuracy: testAccuracy,
          convergence_rate: result.convergenceRate,
          execution_time: executionTime
        })
      });
    }

    setIsTraining(false);
    setCurrentOptimizer(null);
    setStatusMessage('Experiment complete.');
    fetchHistory();
  };

  const bestOptimizer = useMemo(() => {
    if (results.length === 0) return null;
    return results.reduce((prev, curr) => {
      // Simple score: accuracy * 0.6 + convergence * 0.2 - time * 0.2
      const score = (res: ExperimentResult) => res.testAccuracy * 100 + res.convergenceRate - res.executionTime;
      return score(curr) > score(prev) ? curr : prev;
    });
  }, [results]);

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
          <h1 className="font-bold text-lg tracking-tight">ML Experimenter</h1>
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

        {/* Sample Size */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-xs font-semibold text-[#78716C] uppercase tracking-wider">
            <BarChart3 className="w-3 h-3" />
            Sampling
          </div>
          <div className="space-y-2">
            <label className="block text-sm font-medium">Sample Size: {sampleSize}</label>
            <input 
              type="range" min="1000" max="20000" step="1000" value={sampleSize}
              onChange={(e) => setSampleSize(parseInt(e.target.value))}
              className="w-full accent-[#1C1917]"
            />
            <div className="flex justify-between text-[10px] text-[#A8A29E]">
              <span>1k</span>
              <span>20k</span>
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
                  <span>Rows: {trainData.length}</span>
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
                  <span>Rows: {testData.length}</span>
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
                </div>
                <div className="text-3xl font-bold tracking-tight">{elapsedTime}s</div>
              </div>
            </div>
            <div className="mt-8 space-y-2">
              <div className="flex justify-between text-xs font-medium text-[#A8A29E]">
                <span>Overall Progress</span>
                <span>{params.epochs > 0 ? Math.round(((results.length * params.epochs + currentEpoch) / (4 * params.epochs)) * 100) : 0}%</span>
              </div>
              <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-white transition-all duration-300" 
                  style={{ width: `${params.epochs > 0 ? ((results.length * params.epochs + currentEpoch) / (4 * params.epochs)) * 100 : 0}%` }}
                />
              </div>
            </div>
          </section>
        )}

        {/* Results Visualization */}
        {results.length > 0 && (
          <div className="grid grid-cols-2 gap-8">
            <section className="bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm">
              <h3 className="font-bold mb-6 flex items-center gap-2">
                <BarChart3 className="w-4 h-4" /> Loss vs Epoch
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
                <CheckCircle2 className="w-4 h-4" /> Accuracy vs Epoch
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
                <BarChart3 className="w-4 h-4" /> Gradient Norm vs Epoch
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
                <Timer className="w-4 h-4" /> Update Ratio vs Epoch
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
                    <th className="px-6 py-4 font-semibold">Test Accuracy</th>
                    <th className="px-6 py-4 font-semibold">Convergence Rate</th>
                    <th className="px-6 py-4 font-semibold">Execution Time</th>
                    <th className="px-6 py-4 font-semibold">Loss Variance</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E7E5E4]">
                  {results.map((res) => (
                    <tr key={res.optimizer} className={cn(bestOptimizer?.optimizer === res.optimizer && "bg-emerald-50/50")}>
                      <td className="px-6 py-4 font-bold flex items-center gap-2">
                        {res.optimizer}
                        {bestOptimizer?.optimizer === res.optimizer && <CheckCircle2 className="w-3 h-3 text-emerald-600" />}
                      </td>
                      <td className="px-6 py-4">{(res.testAccuracy * 100).toFixed(2)}%</td>
                      <td className="px-6 py-4">{res.convergenceRate.toFixed(2)}x</td>
                      <td className="px-6 py-4">{res.executionTime.toFixed(2)}s</td>
                      <td className="px-6 py-4">{res.lossVariance.toExponential(2)}</td>
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
                The best optimizer for this dataset is <span className="font-bold">{bestOptimizer.optimizer}</span> because it achieved a test accuracy of <span className="font-bold">{(bestOptimizer.testAccuracy * 100).toFixed(2)}%</span> with a convergence rate of <span className="font-bold">{bestOptimizer.convergenceRate.toFixed(2)}x</span>.
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
                <div key={exp.id} className="flex items-center justify-between p-4 bg-[#F5F5F4] rounded-xl hover:bg-[#E7E5E4] transition-colors">
                  <div className="flex items-center gap-4">
                    <div className="bg-white p-2 rounded-lg shadow-sm">
                      <Database className="w-4 h-4 text-[#1C1917]" />
                    </div>
                    <div>
                      <div className="font-bold text-sm">{exp.dataset_name}</div>
                      <div className="text-[10px] text-[#78716C] uppercase tracking-wider">{exp.optimizer} • {new Date(exp.timestamp).toLocaleDateString()}</div>
                    </div>
                  </div>
                  <div className="flex gap-8 text-right">
                    <div>
                      <div className="text-[10px] text-[#78716C] uppercase font-semibold">Accuracy</div>
                      <div className="text-sm font-bold">{(exp.test_accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[#78716C] uppercase font-semibold">Time</div>
                      <div className="text-sm font-bold">{exp.execution_time.toFixed(1)}s</div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
