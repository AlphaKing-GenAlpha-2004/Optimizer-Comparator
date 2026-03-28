import React, { useState, useEffect, useMemo, useRef, Component } from 'react';
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
  Scissors, Download, ChevronRight, Activity, Zap, TrendingDown, Grid3X3, Github, Type, Trash2, RefreshCw, Check, LogIn, LogOut, User as UserIcon
} from 'lucide-react';
import { NeuralNetwork, OptimizerType, ModelParams, ExperimentResult, TrainingMetric } from './ml-engine';
import InfoModal from './components/InfoModal';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { auth, db, googleProvider, signInWithPopup, onAuthStateChanged, collection, addDoc, getDocs, query, where, orderBy, limit, deleteDoc, doc, onSnapshot, User, Timestamp } from './firebase';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}


enum OperationType {
  CREATE = 'create',
  UPDATE = 'update',
  DELETE = 'delete',
  LIST = 'list',
  GET = 'get',
  WRITE = 'write',
}

interface FirestoreErrorInfo {
  error: string;
  operationType: OperationType;
  path: string | null;
  authInfo: {
    userId: string | undefined;
    email: string | null | undefined;
    emailVerified: boolean | undefined;
    isAnonymous: boolean | undefined;
    tenantId: string | null | undefined;
    providerInfo: {
      providerId: string;
      displayName: string | null;
      email: string | null;
      photoUrl: string | null;
    }[];
  }
}

function handleFirestoreError(error: unknown, operationType: OperationType, path: string | null) {
  const errInfo: FirestoreErrorInfo = {
    error: error instanceof Error ? error.message : String(error),
    authInfo: {
      userId: auth.currentUser?.uid,
      email: auth.currentUser?.email,
      emailVerified: auth.currentUser?.emailVerified,
      isAnonymous: auth.currentUser?.isAnonymous,
      tenantId: auth.currentUser?.tenantId,
      providerInfo: auth.currentUser?.providerData.map(provider => ({
        providerId: provider.providerId,
        displayName: provider.displayName,
        email: provider.email,
        photoUrl: provider.photoURL
      })) || []
    },
    operationType,
    path
  }
  console.error('Firestore Error: ', JSON.stringify(errInfo));
  throw new Error(JSON.stringify(errInfo));
}

export class ErrorBoundary extends Component<any, any> {
  state: any;
  props: any;
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: any) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      let errorMessage = "Something went wrong.";
      try {
        const parsed = JSON.parse(this.state.error.message);
        if (parsed.error && parsed.error.includes("insufficient permissions")) {
          errorMessage = "You don't have permission to perform this action. Please make sure you are logged in.";
        }
      } catch (e) {}
      
      return (
        <div className="min-h-screen flex items-center justify-center bg-red-50 p-4">
          <div className="bg-white p-8 rounded-2xl shadow-xl max-w-md w-full border border-red-100">
            <div className="flex items-center gap-3 text-red-600 mb-4">
              <AlertCircle className="w-8 h-8" />
              <h2 className="text-xl font-bold">Application Error</h2>
            </div>
            <p className="text-gray-600 mb-6">{errorMessage}</p>
            <button 
              onClick={() => window.location.reload()}
              className="w-full bg-red-600 text-white py-3 rounded-xl font-bold hover:bg-red-700 transition-colors"
            >
              Reload Application
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  // State
  const [trainData, setTrainData] = useState<any[]>([]);
  const [testData, setTestData] = useState<any[]>([]);
  const [trainFile, setTrainFile] = useState<File | null>(null);
  const [testFile, setTestFile] = useState<File | null>(null);
  const [isSingleDatasetMode, setIsSingleDatasetMode] = useState(true);
  const [splitRatio, setSplitRatio] = useState(0.8);
  const [user, setUser] = useState<User | null>(null);
  const [isAuthReady, setIsAuthReady] = useState(false);
  
  const [features, setFeatures] = useState<string[]>([]);
  const [target, setTarget] = useState<string>('');
  const [trainSampleSize, setTrainSampleSize] = useState<number>(10000);
  const [testSampleSize, setTestSampleSize] = useState<number>(2000);
  const [totalSampleSize, setTotalSampleSize] = useState<number>(12000);
  const [isTextEntryOpen, setIsTextEntryOpen] = useState(false);
  const [textEntryData, setTextEntryData] = useState({ train: '', test: '' });
  
  const [params, setParams] = useState<ModelParams>({
    hiddenSize: 64,
    learningRate: 0.01,
    adamLearningRate: 0.001,
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
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [parseProgress, setParseProgress] = useState(0);
  const [results, setResults] = useState<ExperimentResult[]>([]);
  const [optimizerProgress, setOptimizerProgress] = useState<Record<string, { epoch: number, trainProgress: number, testProgress: number }>>({
    SGD: { epoch: 0, trainProgress: 0, testProgress: 0 },
    Adam: { epoch: 0, trainProgress: 0, testProgress: 0 },
    Adagrad: { epoch: 0, trainProgress: 0, testProgress: 0 },
    RMSProp: { epoch: 0, trainProgress: 0, testProgress: 0 }
  });
  const [elapsedTime, setElapsedTime] = useState(0);
  const [history, setHistory] = useState<any[]>([]);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [currentOptimizer, setCurrentOptimizer] = useState<OptimizerType | null>(null);
  const [currentEpoch, setCurrentEpoch] = useState(0);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (u) => {
      setUser(u);
      setIsAuthReady(true);
    });
    return () => unsubscribe();
  }, []);

  const fetchHistory = () => {
    if (!user) {
      setHistory([]);
      return undefined;
    }
    const q = query(
      collection(db, 'experiments'),
      where('uid', '==', user.uid),
      limit(50)
    );
    return onSnapshot(q, (snapshot) => {
      const data = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() as any }));
      // Sort in memory to avoid index requirement
      data.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
      setHistory(data);
    }, (err) => {
      console.error("History snapshot error:", err);
      if (err.message.includes("index")) {
        console.warn("Firestore index required for history query.");
      }
    });
  };

  const fetchDatasets = () => {
    if (!user) {
      setDatasets([]);
      return undefined;
    }
    const q = query(
      collection(db, 'datasets'),
      where('uid', '==', user.uid)
    );
    return onSnapshot(q, (snapshot) => {
      const data = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() as any }));
      // Sort in memory to avoid index requirement
      data.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
      setDatasets(data);
    }, (err) => {
      console.error("Datasets snapshot error:", err);
    });
  };

  useEffect(() => {
    let unsubHistory: (() => void) | undefined;
    let unsubDatasets: (() => void) | undefined;
    
    if (isAuthReady && user) {
      unsubHistory = fetchHistory();
      unsubDatasets = fetchDatasets();
    }
    
    return () => {
      unsubHistory?.();
      unsubDatasets?.();
    };
  }, [isAuthReady, user]);

  useEffect(() => {
    let interval: any;
    if (isTraining) {
      interval = setInterval(() => {
        setElapsedTime(prev => prev + 1);
      }, 1000);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [isTraining]);

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

  const clearData = () => {
    setTrainData([]);
    setTestData([]);
    setTrainTensors(null);
    setTestTensors(null);
    setTrainFile(null);
    setTestFile(null);
    setFeatures([]);
    setTarget('');
    setResults([]);
    setStatusMessage('Memory cleared.');
  };

  const togglePause = () => {
    setIsPaused(!isPaused);
    isPausedRef.current = !isPausedRef.current;
  };

  const [trainTensors, setTrainTensors] = useState<{ X: Float32Array, y: Int32Array } | null>(null);
  const [testTensors, setTestTensors] = useState<{ X: Float32Array, y: Int32Array } | null>(null);
  const [classes, setClasses] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showWarning, setShowWarning] = useState(false);
  const workersRef = useRef<Worker[]>([]);
  const timeoutRef = useRef<any>(null);

  const getDynamicBatchSize = (samples: number) => {
    if (samples < 10000) return 128;
    if (samples < 50000) return 256;
    return 512;
  };

  const getDynamicEpochLimit = (samples: number) => {
    if (!samples) return 5000;
    // Aim for ~5M total samples processed as a reasonable browser limit
    // but keep a minimum of 100 epochs and maximum of 10000
    return Math.min(10000, Math.max(100, Math.floor(5000000 / samples)));
  };

  const getDynamicHiddenSizeLimit = (inputSize: number) => {
    if (!inputSize) return 512;
    // Allow hidden size up to 2x input size, but at least 512 and at most 2048
    return Math.min(2048, Math.max(512, inputSize * 2));
  };

  const getDynamicSampleSizeLimit = (inputSize: number) => {
    if (!inputSize) return 1000000;
    // 250M elements total limit for Float32Array (~1GB)
    // Capped at 5M for UI responsiveness and stability
    return Math.min(5000000, Math.floor(250000000 / inputSize));
  };

  // Re-process files when sample size sliders are adjusted
  useEffect(() => {
    if (trainFile) {
      processFile(trainFile, 'train');
    }
  }, [trainSampleSize, totalSampleSize, splitRatio, isSingleDatasetMode]);

  useEffect(() => {
    if (testFile && features.length > 0 && !isSingleDatasetMode) {
      processFile(testFile, 'test');
    }
  }, [testSampleSize, features, target, isSingleDatasetMode]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, type: 'train' | 'test') => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (type === 'train') setTrainFile(file);
    else setTestFile(file);

    processFile(file, type);
  };

  const handleLogin = async () => {
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (error) {
      console.error("Login failed:", error);
    }
  };

  const handleLogout = async () => {
    try {
      await auth.signOut();
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  const handleTextEntry = (type: 'train' | 'test') => {
    const blob = new Blob([textEntryData[type]], { type: 'text/csv' });
    const file = new File([blob], `manual_${type}.csv`, { type: 'text/csv' });
    if (type === 'train') setTrainFile(file);
    else setTestFile(file);
    processFile(file, type);
    setIsTextEntryOpen(false);
  };

  const saveDatasetMetadata = async (file: File, type: string, rowCount: number, featCols: string[], targetCol: string) => {
    if (!user) return;
    const path = 'datasets';
    try {
      await addDoc(collection(db, path), {
        name: file.name,
        type: type,
        rowCount: rowCount,
        columnCount: featCols.length,
        features: JSON.stringify(featCols),
        target: targetCol,
        content: '',
        uid: user.uid,
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      handleFirestoreError(err, OperationType.WRITE, path);
    }
  };

  const deleteDataset = async (id: string) => {
    const path = `datasets/${id}`;
    try {
      await deleteDoc(doc(db, 'datasets', id));
    } catch (err) {
      handleFirestoreError(err, OperationType.DELETE, path);
    }
  };

  const loadDatasetMetadata = (ds: any) => {
    setFeatures(JSON.parse(ds.features));
    setTarget(ds.target);
    setStatusMessage(`Loaded metadata for ${ds.name}. Please upload the file to start training.`);
  };

  const processFile = async (file: File, type: 'train' | 'test') => {
    setIsProcessing(true);
    setError(null);
    setParseProgress(0);
    setStatusMessage(`Parsing ${file.name}...`);

    const maxSamples = isSingleDatasetMode 
      ? totalSampleSize 
      : (type === 'train' ? trainSampleSize : testSampleSize);
    const tempRows: any[] = [];
    let rowCount = 0;
    let featCols: string[] = [];
    let targetCol = "";
    
    // Reset classes if it's a new training file to prevent pollution
    // If it's a test file, we MUST use the classes from the training file
    let currentClasses: Set<string> = type === 'train' ? new Set<string>() : new Set<string>(classes);
    
    let X: Float32Array | null = null;
    let y: Int32Array | null = null;

    const rawTargets: string[] = [];

    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      worker: true,
      chunkSize: 1024 * 1024 * 10, // 10MB chunks for better performance with large files
      step: (results, parser) => {
        const row = results.data;
        if (rowCount === 0) {
          const cols = Object.keys(row || {});
          
          if (type === 'test') {
            if (features.length === 0) {
              setError("Please upload a training dataset first to define features and classes.");
              parser.abort();
              setIsProcessing(false);
              return;
            }
            featCols = features;
            targetCol = target || cols[0];
          } else {
            if (cols.length < 2) {
              setError("Dataset must have at least one feature and one target column.");
              parser.abort();
              setIsProcessing(false);
              return;
            }
            targetCol = cols[0];        // first column is label
            featCols = cols.slice(1);   // remaining columns are features
          }
          
          const inputSize = featCols.length;
          
          // Memory Guard: Check if we can allocate the required TypedArrays
          const totalElements = maxSamples * inputSize;
          if (totalElements > 250000000) { // ~1GB limit for safety in browser
            setError(`Dataset too large for browser memory. Reducing samples to ${Math.floor(250000000 / inputSize).toLocaleString()}.`);
            // We'll continue with a smaller maxSamples
          }

          let newHiddenSize;
          if (inputSize <= 1000) {
            newHiddenSize = 64; // MNIST-like
          } else if (inputSize <= 3000) {
            newHiddenSize = 256; // CIFAR-10-like
          } else if (inputSize <= 4000) {
            newHiddenSize = 512; // CIFAR-100-like
          } else {
            newHiddenSize = 1024; // High-res
          }
          
          if (type === 'train') {
            setFeatures(featCols);
            setTarget(targetCol);
            setParams(prev => ({ ...prev, hiddenSize: newHiddenSize }));
          }

          try {
            X = new Float32Array(Math.min(maxSamples, Math.floor(250000000 / inputSize)) * inputSize);
            y = new Int32Array(Math.min(maxSamples, Math.floor(250000000 / inputSize)));
          } catch (e) {
            setError("Memory allocation failed. Try reducing sample size.");
            parser.abort();
            setIsProcessing(false);
            return;
          }
        }

        const effectiveMax = X ? X.length / featCols.length : 0;

        if (rowCount < effectiveMax) {
          // Store first 10 rows for preview
          if (rowCount < 10) {
            const previewRow: any = {};
            const previewCols = featCols.slice(0, 20);
            previewCols.forEach(c => previewRow[c] = row[c]);
            previewRow[targetCol] = row[targetCol];
            tempRows.push(previewRow);
          }

          // Process for TypedArrays
          featCols.forEach((f, j) => {
            let val = parseFloat(row[f]);
            if (!Number.isFinite(val)) val = 0;
            // Auto-normalization check: if values are > 1, assume 0-255 range
            // This is a heuristic, but common for image datasets
            if (val > 1) val /= 255;
            if (X) X[rowCount * featCols.length + j] = val;
          });

          const targetVal = String(row[targetCol]);
          if (type === 'train') {
            currentClasses.add(targetVal);
          }
          rawTargets.push(targetVal);
        }

        rowCount++;
        if (rowCount % 5000 === 0) {
          const progress = Math.min(99, (rowCount / maxSamples) * 100);
          setParseProgress(progress);
          setStatusMessage(`Streaming ${file.name}... ${rowCount.toLocaleString()} rows (${(rowCount * featCols.length * 4 / (1024*1024)).toFixed(1)} MB in memory)`);
        }
        
        if (rowCount >= effectiveMax) {
          parser.abort();
        }
      },
      complete: () => {
        setParseProgress(100);
        if (rowCount === 0) {
          setError("The dataset is empty.");
          setIsProcessing(false);
          return;
        }

        let finalClasses: string[] = [];
        if (type === 'train') {
          finalClasses = Array.from(currentClasses).sort();
          setClasses(finalClasses);
        } else {
          finalClasses = classes;
          if (finalClasses.length === 0) {
            setError("Please upload a training dataset first to define the classes.");
            setIsProcessing(false);
            return;
          }
        }

        const classMap = new Map(finalClasses.map((c, i) => [c, i]));

        // Map raw targets to indices
        if (y) {
          rawTargets.forEach((t, i) => {
            y![i] = classMap.has(t) ? classMap.get(t)! : 0;
          });
        }

        // Fisher-Yates Shuffle to prevent class imbalance from sorted datasets
        // Optimized for large datasets to avoid UI freeze
        if (X && y && rowCount > 0) {
          const featCount = featCols.length;
          const shuffleLimit = Math.min(rowCount, 100000); // Limit shuffle for very large datasets to keep UI responsive
          for (let i = shuffleLimit - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            
            const tempY = y[i];
            y[i] = y[j];
            y[j] = tempY;
            
            for (let k = 0; k < featCount; k++) {
              const tempX = X[i * featCount + k];
              X[i * featCount + k] = X[j * featCount + k];
              X[j * featCount + k] = tempX;
            }
          }
        }

        if (type === 'train') {
          setTrainData(tempRows);
          if (X && y) {
            if (isSingleDatasetMode) {
              const trainCount = Math.floor(rowCount * splitRatio);
              const testCount = rowCount - trainCount;
              const featCount = featCols.length;

              setTrainTensors({
                X: X.subarray(0, trainCount * featCount),
                y: y.subarray(0, trainCount)
              });
              setTestTensors({
                X: X.subarray(trainCount * featCount, rowCount * featCount),
                y: y.subarray(trainCount, rowCount)
              });
              setTestData(tempRows.slice(Math.floor(tempRows.length * splitRatio)));
              setTrainData(tempRows.slice(0, Math.floor(tempRows.length * splitRatio)));
              setStatusMessage(`Dataset split: ${trainCount.toLocaleString()} train, ${testCount.toLocaleString()} test.`);
            } else {
              setTrainTensors({ 
                X: X.subarray(0, rowCount * featCols.length), 
                y: y.subarray(0, rowCount) 
              });
            }
          }
          if (rowCount >= 100000) setShowWarning(true);
        } else {
          setTestData(tempRows);
          if (X && y) {
            setTestTensors({ 
              X: X.subarray(0, rowCount * featCols.length), 
              y: y.subarray(0, rowCount) 
            });
          }
        }

        setIsProcessing(false);
        setStatusMessage(`${file.name} processed (${rowCount} rows).`);
        saveDatasetMetadata(file, type, rowCount, featCols, targetCol);
      }
    });
  };

  const startTraining = async () => {
    if (!trainTensors || !testTensors || !target) {
      setError("Please upload both training and testing datasets first.");
      return;
    }

    setIsTraining(true);
    setError(null);

    if (params.learningRate <= 0 || params.learningRate > 1 || params.adamLearningRate <= 0 || params.adamLearningRate > 1) {
      setError("Learning rates must be between 0 and 1.");
      setIsTraining(false);
      return;
    }

    // Dimension Check (Step 12)
    const trainInputSize = trainTensors.X.length / trainTensors.y.length;
    const testInputSize = testTensors.X.length / testTensors.y.length;
    const numSamples = trainTensors.y.length;
    
    const epochLimit = getDynamicEpochLimit(numSamples);
    const hiddenLimit = getDynamicHiddenSizeLimit(trainInputSize);

    if (params.hiddenSize < 1 || params.hiddenSize > hiddenLimit) {
      setError(`Hidden layer size must be between 1 and ${hiddenLimit} for this dataset.`);
      setIsTraining(false);
      return;
    }

    if (params.epochs < 1 || params.epochs > epochLimit) {
      setError(`Number of epochs must be between 1 and ${epochLimit} for this dataset size.`);
      setIsTraining(false);
      return;
    }
    
    if (trainInputSize !== features.length || testInputSize !== features.length) {
      setError(`Dimension mismatch: Train features (${trainInputSize}) vs Test features (${testInputSize}) vs Expected (${features.length}). Please re-upload datasets.`);
      setIsTraining(false);
      return;
    }

    setIsPaused(false);
    isPausedRef.current = false;
    stopTrainingRef.current = false;
    setResults([]);
    setElapsedTime(0);
    setOptimizerProgress({
      SGD: { epoch: 0, trainProgress: 0, testProgress: 0 },
      Adam: { epoch: 0, trainProgress: 0, testProgress: 0 },
      Adagrad: { epoch: 0, trainProgress: 0, testProgress: 0 },
      RMSProp: { epoch: 0, trainProgress: 0, testProgress: 0 }
    });

    const optimizers: OptimizerType[] = ['SGD', 'Adagrad', 'RMSProp', 'Adam'];
    const allResults: ExperimentResult[] = [];
    const cores = navigator.hardwareConcurrency || 4;
    let batchSize = params.batchSize || getDynamicBatchSize(trainTensors.y.length);
    let epochs = params.epochs;

    // Auto Performance Mode (Step 9)
    if (trainTensors.y.length > 10000) {
      batchSize = 128;
      if (trainTensors.y.length > 50000) {
        batchSize = 256;
        epochs = Math.min(epochs, 8);
      }
      setStatusMessage(`Auto Performance Mode: Batch size ${batchSize}, Epochs ${epochs}`);
    } else {
      setStatusMessage(`Starting parallel training on ${cores} cores...`);
    }

    // Global timeout safety (60,000 seconds)
    const MAX_TRAINING_TIME = 60000000;
    timeoutRef.current = setTimeout(() => {
      stopTraining();
      setStatusMessage("Training stopped due to computational limits (60000s).");
    }, MAX_TRAINING_TIME);

    const runOptimizer = (opt: OptimizerType): Promise<ExperimentResult> => {
      return new Promise((resolve, reject) => {
        const worker = new Worker(new URL('./training-worker.ts', import.meta.url), { type: 'module' });
        workersRef.current.push(worker);

        // Per-worker watchdog timer
        let watchdog: any;
        const resetWatchdog = () => {
          if (watchdog) clearTimeout(watchdog);
          watchdog = setTimeout(() => {
            worker.terminate();
            reject(new Error(`Optimizer ${opt} timed out (no progress for 1200s).`));
          }, 1200000); // Increased to 1200 seconds (20 minutes) of inactivity
        };

        resetWatchdog();

        worker.onmessage = (e) => {
          resetWatchdog();
          const { type, optimizer, epoch, trainProgress, testProgress, metrics: result } = e.data;
          
          if (type === 'progress') {
            setOptimizerProgress(prev => ({
              ...prev,
              [optimizer]: { epoch, trainProgress, testProgress }
            }));
          } else if (type === 'training_complete') {
            clearTimeout(watchdog);
            worker.terminate();
            resolve(result);
          } else if (type === 'timeout') {
            clearTimeout(watchdog);
            worker.terminate();
            reject(new Error(`Optimizer ${optimizer} timed out.`));
          } else if (type === 'error') {
            clearTimeout(watchdog);
            worker.terminate();
            const errMsg = e.data.message || `An error occurred in ${optimizer} worker.`;
            setError(errMsg);
            reject(new Error(errMsg));
          }
        };

        worker.onerror = (err) => {
          console.error("Worker Error:", err);
          clearTimeout(watchdog);
          worker.terminate();
          const errorMessage = err.message || "Unknown worker error (possible memory limit or crash)";
          reject(new Error(`Worker crash: ${errorMessage}`));
        };

        // Clone buffers for each worker to avoid detaching the original ones (Step 7)
        // This is critical for parallel execution as transferring a buffer detaches it.
        const X_train_buf = new Float32Array(trainTensors.X);
        const y_train_buf = new Int32Array(trainTensors.y);
        const X_test_buf = new Float32Array(testTensors.X);
        const y_test_buf = new Int32Array(testTensors.y);

        // Optimizer-Specific Learning Rates
        let lr = (opt === 'Adam' || opt === 'RMSProp') ? params.adamLearningRate : params.learningRate;

        worker.postMessage({
          optimizer: opt,
          hiddenSize: params.hiddenSize,
          learningRate: lr,
          epochs: epochs,
          batchSize: batchSize,
          inputSize: features.length,
          outputSize: classes.length,
          X_train: X_train_buf,
          y_train: y_train_buf,
          X_test: X_test_buf,
          y_test: y_test_buf,
          trainSamples: trainTensors.y.length,
          testSamples: testTensors.y.length
        }, [X_train_buf.buffer, y_train_buf.buffer, X_test_buf.buffer, y_test_buf.buffer]);
      });
    };

    // Execution Strategy: Spawn all simultaneously (Step 6)
    setStatusMessage(`Spawning all optimizers in parallel...`);
    
    try {
      const results = await Promise.all(optimizers.map(opt => runOptimizer(opt)));
      setResults(results);
      
      // Save all results to DB
      if (user) {
        const path = 'experiments';
        console.log(`Saving ${results.length} experiments to Firestore...`);
        for (const res of results) {
          try {
            await addDoc(collection(db, path), {
              dataset_name: trainFile?.name || 'Manual Entry',
              sample_size: trainTensors.y.length,
              train_test_split: (trainTensors.y.length / (trainTensors.y.length + testTensors.y.length)) * 100,
              optimizer: res.optimizer,
              hidden_size: params.hiddenSize,
              learning_rate: res.learningRate,
              epochs: params.epochs,
              batch_size: batchSize,
              test_accuracy: res.testAccuracy,
              precision: res.precision,
              recall: res.recall,
              f1_score: res.f1Score,
              log_loss: res.logLoss,
              convergence_rate: res.convergenceRate,
              training_time: res.trainingTime,
              testing_time: res.testingTime,
              execution_time: res.executionTime,
              aulc: res.aulc,
              loss_variance: res.lossVariance,
              convergence_epoch_85: res.convergenceEpoch85 || null,
              avg_gradient_norm: res.avgGradientNorm,
              avg_gradient_variance: res.avgGradientVariance,
              avg_throughput: res.avgThroughput,
              avg_update_ratio: res.avgUpdateRatio,
              logs: JSON.stringify(res.metrics),
              uid: user.uid,
              timestamp: new Date().toISOString()
            });
          } catch (err) {
            handleFirestoreError(err, OperationType.WRITE, path);
          }
        }
      }
      
      setStatusMessage("Parallel training complete.");
    } catch (err: any) {
      console.error("Training failed:", err);
      setError(err.message);
    } finally {
      setIsTraining(false);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      fetchHistory();
    }
  };

  const bestOptimizer = useMemo(() => {
    if (results.length === 0) return null;

    // Normalization helper to bring all metrics to [0, 1] range
    const normalize = (val: number, min: number, max: number, invert = false) => {
      if (max === min) return 1.0; // If all values are the same, they all get full score for this metric
      const norm = (val - min) / (max - min);
      return invert ? 1.0 - norm : norm;
    };

    // Extract all metrics for boundary calculation
    const metrics = {
      testAccuracy: results.map(r => r.testAccuracy),
      trainAccuracy: results.map(r => r.metrics[r.metrics.length - 1]?.trainAccuracy || 0),
      trainLoss: results.map(r => r.metrics[r.metrics.length - 1]?.loss || 10),
      f1Score: results.map(r => r.f1Score),
      logLoss: results.map(r => r.logLoss),
      convergenceRate: results.map(r => r.convergenceRate),
      convergenceEpoch85: results.map(r => r.convergenceEpoch85 || (params.epochs + 1)),
      trainingTime: results.map(r => r.trainingTime),
      lossVariance: results.map(r => r.lossVariance),
      aulc: results.map(r => r.aulc),
      avgGradientNorm: results.map(r => r.avgGradientNorm),
      avgUpdateRatio: results.map(r => r.avgUpdateRatio)
    };

    const bounds = {
      testAccuracy: { min: Math.min(...metrics.testAccuracy), max: Math.max(...metrics.testAccuracy) },
      trainAccuracy: { min: Math.min(...metrics.trainAccuracy), max: Math.max(...metrics.trainAccuracy) },
      trainLoss: { min: Math.min(...metrics.trainLoss), max: Math.max(...metrics.trainLoss) },
      f1Score: { min: Math.min(...metrics.f1Score), max: Math.max(...metrics.f1Score) },
      logLoss: { min: Math.min(...metrics.logLoss), max: Math.max(...metrics.logLoss) },
      convergenceRate: { min: Math.min(...metrics.convergenceRate), max: Math.max(...metrics.convergenceRate) },
      convergenceEpoch85: { min: Math.min(...metrics.convergenceEpoch85), max: Math.max(...metrics.convergenceEpoch85) },
      trainingTime: { min: Math.min(...metrics.trainingTime), max: Math.max(...metrics.trainingTime) },
      lossVariance: { min: Math.min(...metrics.lossVariance), max: Math.max(...metrics.lossVariance) },
      aulc: { min: Math.min(...metrics.aulc), max: Math.max(...metrics.aulc) },
      avgGradientNorm: { min: Math.min(...metrics.avgGradientNorm), max: Math.max(...metrics.avgGradientNorm) },
      avgUpdateRatio: { min: Math.min(...metrics.avgUpdateRatio), max: Math.max(...metrics.avgUpdateRatio) }
    };

    // Calculate composite score for each result
    // Weights are distributed based on user-requested metrics:
    // Test Accuracy (20%), Training Accuracy (15%), Training Loss (15%), 
    // Convergence Rate (15%), Convergence Epoch 85% (15%), Gradient Norm (10%),
    // F1 Score (5%), Stability/Var (5%)
    const scored = results.map(res => {
      const lastMetric = res.metrics[res.metrics.length - 1];
      const trainAcc = lastMetric?.trainAccuracy || 0;
      const trainLoss = lastMetric?.loss || 10;
      const convEpoch = res.convergenceEpoch85 || (params.epochs + 1);

      const sTestAcc = normalize(res.testAccuracy, bounds.testAccuracy.min, bounds.testAccuracy.max) * 0.20;
      const sTrainAcc = normalize(trainAcc, bounds.trainAccuracy.min, bounds.trainAccuracy.max) * 0.15;
      const sTrainLoss = normalize(trainLoss, bounds.trainLoss.min, bounds.trainLoss.max, true) * 0.15;
      const sConvRate = normalize(res.convergenceRate, bounds.convergenceRate.min, bounds.convergenceRate.max) * 0.15;
      const sConvEpoch = normalize(convEpoch, bounds.convergenceEpoch85.min, bounds.convergenceEpoch85.max, true) * 0.15;
      const sGradNorm = normalize(res.avgGradientNorm, bounds.avgGradientNorm.min, bounds.avgGradientNorm.max, true) * 0.10;
      const sF1 = normalize(res.f1Score, bounds.f1Score.min, bounds.f1Score.max) * 0.05;
      const sVar = normalize(res.lossVariance, bounds.lossVariance.min, bounds.lossVariance.max, true) * 0.05;

      return {
        ...res,
        compositeScore: sTestAcc + sTrainAcc + sTrainLoss + sConvRate + sConvEpoch + sGradNorm + sF1 + sVar
      };
    });

    // Sort by composite score descending
    return scored.sort((a, b) => b.compositeScore - a.compositeScore)[0];
  }, [results]);

  const deleteExperiment = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    // If we're not already confirming this one, set it to confirm
    if (confirmDeleteId !== id) {
      setConfirmDeleteId(id);
      // Auto-reset after 3 seconds if not clicked again
      setTimeout(() => setConfirmDeleteId(null), 3000);
      return;
    }

    try {
      await deleteDoc(doc(db, 'experiments', id));
      setConfirmDeleteId(null);
      if (selectedExperiment?.id === id) {
        setIsViewingReport(false);
        setSelectedExperiment(null);
      }
    } catch (err) {
      handleFirestoreError(err, OperationType.DELETE, `experiments/${id}`);
    }
  };

  const fetchExperimentDetails = (id: string) => {
    const exp = history.find(h => h.id === id);
    if (exp) {
      const data = { ...exp };
      if (typeof data.logs === 'string') {
        try {
          data.logs = JSON.parse(data.logs);
        } catch (e) {
          console.error("Failed to parse logs:", e);
        }
      }
      setSelectedExperiment(data);
      setIsViewingReport(true);
    }
  };

  const calculateANOVA = (results: ExperimentResult[]) => {
    if (results.length < 2) return null;

    // We'll compare the final accuracies of the optimizers
    // Since we only have 1 run per optimizer, we can't do a standard ANOVA on final accuracy.
    // Instead, let's treat the last 5 epochs of each optimizer as samples to see if they've stabilized at different levels.
    const groups = results.map(res => {
      const lastEpochs = res.metrics.slice(-5).map(m => m.accuracy);
      return lastEpochs;
    }).filter(g => g.length > 0);

    if (groups.length < 2) return null;

    const k = groups.length; 
    const n = groups[0].length; 
    const N = groups.reduce((sum, g) => sum + g.length, 0);

    const groupMeans = groups.map(g => g.reduce((a, b) => a + b, 0) / g.length);
    const grandMean = groups.flat().reduce((a, b) => a + b, 0) / N;

    const ssBetween = groups.reduce((sum, g, i) => sum + g.length * Math.pow(groupMeans[i] - grandMean, 2), 0);
    const ssWithin = groups.reduce((sum, g, i) => {
      return sum + g.reduce((s, val) => s + Math.pow(val - groupMeans[i], 2), 0);
    }, 0);

    const dfBetween = k - 1;
    const dfWithin = N - k;

    const msBetween = ssBetween / dfBetween;
    const msWithin = ssWithin / dfWithin;

    const fValue = msBetween / (msWithin || 1e-10);
    
    // Simple p-value approximation for F-distribution
    // This is a very rough approximation for demonstration
    const pValue = fValue > 4.0 ? 0.001 : (fValue > 2.5 ? 0.05 : 0.5);

    return {
      fValue,
      pValue,
      ssBetween,
      ssWithin,
      dfBetween,
      dfWithin,
      msBetween,
      msWithin
    };
  };

  const anovaResult = useMemo(() => calculateANOVA(results), [results]);

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
      ["Loss Variance", safeFixed(exp.loss_variance, 6)],
      ["Gradient Variance", safeFixed(exp.logs[exp.logs.length - 1].gradientVariance, 6)],
      ["Parameter Norm", safeFixed(exp.logs[exp.logs.length - 1].parameterNorm, 4)],
      ["Throughput", `${safeFixed(exp.logs[exp.logs.length - 1].throughput, 1)} samples/s`]
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

    // ANOVA Table (Only in Report)
    if (results.length >= 2) {
      const anova = calculateANOVA(results);
      if (anova) {
        doc.addPage();
        doc.setFontSize(16);
        doc.text("4. Statistical Analysis (ANOVA)", 14, 20);
        doc.setFontSize(10);
        doc.text("Comparing the stability of optimizers over the final 5 epochs.", 14, 28);

        const anovaRows = [
          ["Between Groups", anova.dfBetween, safeFixed(anova.ssBetween, 6), safeFixed(anova.msBetween, 6), safeFixed(anova.fValue, 4)],
          ["Within Groups", anova.dfWithin, safeFixed(anova.ssWithin, 6), safeFixed(anova.msWithin, 6), ""],
          ["Total", anova.dfBetween + anova.dfWithin, safeFixed(anova.ssBetween + anova.ssWithin, 6), "", ""]
        ];

        autoTable(doc, {
          startY: 35,
          head: [["Source", "DF", "SS", "MS", "F"]],
          body: anovaRows,
          theme: 'grid',
          headStyles: { fillColor: [59, 130, 246] }
        });
      }
    }

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
      { id: 'report-chart-train-test', title: 'Train vs Test Accuracy' },
      { id: 'report-chart-grads', title: 'Gradients & Updates' },
      { id: 'report-chart-speed', title: 'Convergence Speed' },
      { id: 'report-chart-stability', title: 'Training Stability' },
      { id: 'report-chart-advanced-1', title: 'Gradient & Loss Variance' },
      { id: 'report-chart-advanced-2', title: 'Parameter Norm & Throughput' }
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
      safeFixed(m.trainAccuracy || m.accuracy, 2, 100, '%'),
      safeFixed(m.testAccuracy || m.accuracy, 2, 100, '%'),
      safeFixed(m.gradientNorm, 4),
      safeFixed(m.updateRatio, 6)
    ]);

    autoTable(doc, {
      startY: 30,
      head: [["Epoch", "Loss", "Train Acc", "Test Acc", "Grad Norm", "Update Ratio"]],
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
        if (res.metrics[i]) {
          entry[`${res.optimizer}_loss`] = res.metrics[i].loss;
          entry[`${res.optimizer}_acc`] = res.metrics[i].accuracy;
          entry[`${res.optimizer}_train_acc`] = res.metrics[i].trainAccuracy || res.metrics[i].accuracy;
          entry[`${res.optimizer}_test_acc`] = res.metrics[i].testAccuracy || res.metrics[i].accuracy;
          entry[`${res.optimizer}_grad`] = res.metrics[i].gradientNorm;
          entry[`${res.optimizer}_ratio`] = res.metrics[i].updateRatio;
        }
      });
      data.push(entry);
    }
    return data;
  }, [results]);

  const overallProgress = useMemo(() => {
    const total = (Object.values(optimizerProgress) as any[]).reduce((acc: number, curr: any) => {
      return acc + (curr.trainProgress + curr.testProgress) / 2;
    }, 0);
    return (total as number) / 4;
  }, [optimizerProgress]);

  return (
    <div className="min-h-screen bg-[#F5F5F4] text-[#1C1917] font-sans flex">
      {/* Sidebar */}
      <aside className="w-80 bg-white border-r border-[#E7E5E4] p-6 flex flex-col gap-8 overflow-y-auto">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-[#1C1917] p-2 rounded-lg">
              <Database className="text-white w-5 h-5" />
            </div>
            <div className="flex items-center gap-2">
              <h1 className="font-bold text-lg tracking-tight">Neur-O-Opt Lab</h1>
              <button 
                onClick={() => setIsHelpOpen(true)}
                className="p-1 hover:bg-[#F5F5F4] rounded-full transition-colors group relative"
                title="Learn about Optimizers"
              >
                <Info className="w-4 h-4 text-[#78716C] group-hover:text-emerald-600" />
              </button>
              <a 
                href="https://github.com/AlphaKing-GenAlpha-2004/Optimizer-Comparator.git"
                target="_blank"
                rel="noopener noreferrer"
                className="p-1 hover:bg-[#F5F5F4] rounded-full transition-colors group relative"
                title="View on GitHub"
              >
                <Github className="w-4 h-4 text-[#78716C] group-hover:text-[#1C1917]" />
              </a>
            </div>
          </div>
          {isAuthReady && (
            user ? (
              <div className="flex items-center gap-2">
                {user.photoURL && <img src={user.photoURL} alt="User" className="w-6 h-6 rounded-full border border-[#E7E5E4]" referrerPolicy="no-referrer" />}
                <button onClick={handleLogout} className="p-2 hover:bg-red-50 rounded-full transition-colors group" title="Logout">
                  <LogOut className="w-4 h-4 text-[#78716C] group-hover:text-red-600" />
                </button>
              </div>
            ) : (
              <button onClick={handleLogin} className="p-2 hover:bg-emerald-50 rounded-full transition-colors group" title="Login with Google">
                <LogIn className="w-4 h-4 text-[#78716C] group-hover:text-emerald-600" />
              </button>
            )
          )}
        </div>

        {/* Dataset Upload */}
        <section className="space-y-4">
          <div className="flex items-center justify-between text-xs font-semibold text-[#78716C] uppercase tracking-wider">
            <div className="flex items-center gap-2">
              <Upload className="w-3 h-3" />
              Dataset Entry
            </div>
            <button 
              onClick={() => setIsTextEntryOpen(!isTextEntryOpen)}
              className="text-[10px] hover:text-[#1C1917] transition-colors flex items-center gap-1"
            >
              <Type className="w-3 h-3" />
              {isTextEntryOpen ? 'Hide Text Entry' : 'Text Entry'}
            </button>
          </div>

          <div className="flex items-center justify-between p-2 bg-[#F5F5F4] rounded-xl">
            <span className="text-[10px] font-bold text-[#78716C] uppercase">Single Dataset Mode (Auto Split)</span>
            <button 
              onClick={() => {
                const newMode = !isSingleDatasetMode;
                if (newMode) {
                  setTotalSampleSize(trainSampleSize + testSampleSize);
                } else {
                  setTrainSampleSize(Math.floor(totalSampleSize * splitRatio));
                  setTestSampleSize(totalSampleSize - Math.floor(totalSampleSize * splitRatio));
                }
                setIsSingleDatasetMode(newMode);
              }}
              className={cn(
                "w-10 h-5 rounded-full transition-colors relative",
                isSingleDatasetMode ? "bg-emerald-500" : "bg-[#E7E5E4]"
              )}
            >
              <div className={cn(
                "absolute top-1 w-3 h-3 bg-white rounded-full transition-all",
                isSingleDatasetMode ? "left-6" : "left-1"
              )} />
            </button>
          </div>

          {isSingleDatasetMode && (
            <div className="space-y-2">
              <div className="flex justify-between items-center text-[11px] font-medium">
                <span>Train-Test Split</span>
                <span className="text-[10px] text-[#78716C] font-bold">{(splitRatio * 100).toFixed(0)}% / {((1 - splitRatio) * 100).toFixed(0)}%</span>
              </div>
              <input 
                type="range" min="0.5" max="0.95" step="0.05" value={splitRatio}
                onChange={(e) => setSplitRatio(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-[#F5F5F4] rounded-lg appearance-none cursor-pointer accent-[#1C1917]"
              />
            </div>
          )}

          <div className="p-3 bg-blue-50 border border-blue-100 rounded-xl">
            <div className="flex items-center gap-2 text-blue-800 font-bold text-[10px] uppercase mb-1">
              <Info className="w-3 h-3" />
              Supported Formats
            </div>
            <p className="text-[9px] text-blue-700 leading-tight">
              Supports CSV/TXT files with labels in the 1st column. Perfect for <b>MNIST</b> (784 features) and <b>CIFAR-100</b> (3072 features).
            </p>
          </div>
          
          {isTextEntryOpen ? (
            <div className="space-y-4 animate-in fade-in slide-in-from-top-2 duration-300">
              <div className="space-y-2">
                <label className="block text-[11px] font-bold text-[#78716C] uppercase">{isSingleDatasetMode ? 'Dataset' : 'Training Data'} (CSV Format)</label>
                <textarea 
                  value={textEntryData.train}
                  onChange={(e) => setTextEntryData({...textEntryData, train: e.target.value})}
                  placeholder="label,feat1,feat2...&#10;0,0.5,0.2...&#10;1,0.1,0.8..."
                  className="w-full h-32 bg-[#F5F5F4] border-none rounded-xl p-3 text-xs font-mono focus:ring-1 ring-[#1C1917] resize-none"
                />
                <button 
                  onClick={() => handleTextEntry('train')}
                  disabled={!textEntryData.train.trim()}
                  className="w-full py-2 bg-[#1C1917] text-white text-[10px] font-bold rounded-lg hover:bg-black disabled:opacity-50 transition-all"
                >
                  Process {isSingleDatasetMode ? 'Dataset' : 'Training Text'}
                </button>
              </div>
              {!isSingleDatasetMode && (
                <div className="space-y-2">
                  <label className="block text-[11px] font-bold text-[#78716C] uppercase">Testing Data (CSV Format)</label>
                  <textarea 
                    value={textEntryData.test}
                    onChange={(e) => setTextEntryData({...textEntryData, test: e.target.value})}
                    placeholder="label,feat1,feat2...&#10;0,0.4,0.3...&#10;1,0.2,0.7..."
                    className="w-full h-32 bg-[#F5F5F4] border-none rounded-xl p-3 text-xs font-mono focus:ring-1 ring-[#1C1917] resize-none"
                  />
                  <button 
                    onClick={() => handleTextEntry('test')}
                    disabled={!textEntryData.test.trim()}
                    className="w-full py-2 bg-[#1C1917] text-white text-[10px] font-bold rounded-lg hover:bg-black disabled:opacity-50 transition-all"
                  >
                    Process Testing Text
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium mb-1">{isSingleDatasetMode ? 'Dataset CSV' : 'Training CSV'}</label>
                <input 
                  type="file" accept=".csv,.txt" 
                  onChange={(e) => handleFileUpload(e, 'train')}
                  className="block w-full text-xs text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-[#F5F5F4] file:text-[#1C1917] hover:file:bg-[#E7E5E4] cursor-pointer"
                />
                {trainFile && <div className="mt-1 text-[10px] text-emerald-600 font-bold">Loaded: {trainFile.name} ({(trainFile.size / (1024*1024)).toFixed(2)} MB)</div>}
              </div>
              {!isSingleDatasetMode && (
                <div>
                  <label className="block text-sm font-medium mb-1">Testing CSV</label>
                  <input 
                    type="file" accept=".csv,.txt" 
                    onChange={(e) => handleFileUpload(e, 'test')}
                    className="block w-full text-xs text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-[#F5F5F4] file:text-[#1C1917] hover:file:bg-[#E7E5E4] cursor-pointer"
                  />
                  {testFile && <div className="mt-1 text-[10px] text-emerald-600 font-bold">Loaded: {testFile.name} ({(testFile.size / (1024*1024)).toFixed(2)} MB)</div>}
                </div>
              )}
            </div>
          )}
        </section>

        {/* Sample Sizes */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-xs font-semibold text-[#78716C] uppercase tracking-wider">
            <BarChart3 className="w-3 h-3" />
            Sampling & Large File Support
          </div>
          <div className="space-y-4">
            {isSingleDatasetMode ? (
              <div className="space-y-2">
                <div className="flex justify-between items-center text-[11px] font-medium">
                  <span>Total Samples</span>
                  <div className="flex items-center gap-1">
                    <input 
                      type="number"
                      min="100"
                      max={getDynamicSampleSizeLimit(features.length)}
                      value={totalSampleSize}
                      onChange={(e) => setTotalSampleSize(Math.max(100, parseInt(e.target.value) || 0))}
                      className="w-20 px-1 py-0.5 text-right bg-transparent border-b border-[#E7E5E4] focus:border-[#1C1917] outline-none text-[#78716C] text-[10px]"
                    />
                    <span className="text-[10px] text-[#A8A29E]">{totalSampleSize >= 1000000 ? `${(totalSampleSize/1000000).toFixed(1)}M` : ''}</span>
                  </div>
                </div>
                <input 
                  type="range" min="100" max={getDynamicSampleSizeLimit(features.length)} step="100" value={totalSampleSize}
                  onChange={(e) => setTotalSampleSize(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-[#F5F5F4] rounded-lg appearance-none cursor-pointer accent-[#1C1917]"
                />
                <div className="flex justify-between text-[10px] text-[#A8A29E]">
                  <span>100</span>
                  <span>{getDynamicSampleSizeLimit(features.length) >= 1000000 ? `${(getDynamicSampleSizeLimit(features.length)/1000000).toFixed(1)}M` : (getDynamicSampleSizeLimit(features.length)/1000).toFixed(0) + 'k'}</span>
                </div>
              </div>
            ) : (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-[11px] font-medium">
                    <span>Training Samples</span>
                  <div className="flex items-center gap-1">
                    <input 
                      type="number"
                      min="50"
                      max={getDynamicSampleSizeLimit(features.length)}
                      value={trainSampleSize}
                      onChange={(e) => setTrainSampleSize(Math.max(50, parseInt(e.target.value) || 0))}
                      className="w-20 px-1 py-0.5 text-right bg-transparent border-b border-[#E7E5E4] focus:border-[#1C1917] outline-none text-[#78716C] text-[10px]"
                    />
                    <span className="text-[10px] text-[#A8A29E]">{trainSampleSize >= 1000000 ? `${(trainSampleSize/1000000).toFixed(1)}M` : ''}</span>
                  </div>
                </div>
                <input 
                  type="range" min="50" max={getDynamicSampleSizeLimit(features.length)} step="50" value={trainSampleSize}
                  onChange={(e) => setTrainSampleSize(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-[#F5F5F4] rounded-lg appearance-none cursor-pointer accent-[#1C1917]"
                />
                <div className="flex justify-between text-[10px] text-[#A8A29E]">
                  <span>50</span>
                  <span>{getDynamicSampleSizeLimit(features.length) >= 1000000 ? `${(getDynamicSampleSizeLimit(features.length)/1000000).toFixed(1)}M` : (getDynamicSampleSizeLimit(features.length)/1000).toFixed(0) + 'k'}</span>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center text-[11px] font-medium">
                  <span>Testing Samples</span>
                  <div className="flex items-center gap-1">
                    <input 
                      type="number"
                      min="20"
                      max={Math.floor(getDynamicSampleSizeLimit(features.length) / 2)}
                      value={testSampleSize}
                      onChange={(e) => setTestSampleSize(Math.max(20, parseInt(e.target.value) || 0))}
                      className="w-20 px-1 py-0.5 text-right bg-transparent border-b border-[#E7E5E4] focus:border-[#1C1917] outline-none text-[#78716C] text-[10px]"
                    />
                    <span className="text-[10px] text-[#A8A29E]">{testSampleSize >= 1000000 ? `${(testSampleSize/1000000).toFixed(1)}M` : ''}</span>
                  </div>
                </div>
                <input 
                  type="range" min="20" max={Math.floor(getDynamicSampleSizeLimit(features.length) / 2)} step="10" value={testSampleSize}
                  onChange={(e) => setTestSampleSize(parseInt(e.target.value))}
                  className="w-full h-1.5 bg-[#F5F5F4] rounded-lg appearance-none cursor-pointer accent-[#1C1917]"
                />
                <div className="flex justify-between text-[10px] text-[#A8A29E]">
                  <span>20</span>
                  <span>{Math.floor(getDynamicSampleSizeLimit(features.length) / 2) >= 1000000 ? `${(Math.floor(getDynamicSampleSizeLimit(features.length) / 2)/1000000).toFixed(1)}M` : (Math.floor(getDynamicSampleSizeLimit(features.length) / 2)/1000).toFixed(0) + 'k'}</span>
                </div>
                </div>
              </>
            )}
            {(trainFile?.size || 0) > 500 * 1024 * 1024 && (
              <div className="p-3 bg-amber-50 border border-amber-100 rounded-xl">
                <div className="flex items-center gap-2 text-amber-800 font-bold text-[10px] uppercase mb-1">
                  <AlertCircle className="w-3 h-3" />
                  Large File Detected
                </div>
                <p className="text-[9px] text-amber-700 leading-tight">
                  Files over 500MB are processed using streaming. To prevent memory issues, we recommend keeping samples under 500k.
                </p>
              </div>
            )}
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
                type="number" min="1" max={getDynamicHiddenSizeLimit(features.length)} value={params.hiddenSize || ''} 
                onChange={e => {
                  const val = parseInt(e.target.value);
                  setParams({...params, hiddenSize: isNaN(val) ? 0 : val});
                }}
                className="w-full bg-[#F5F5F4] border-none rounded-md px-3 py-2 text-sm focus:ring-1 ring-[#1C1917]"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[11px] font-medium">Base LR (SGD/Adagrad)</label>
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
              <label className="text-[11px] font-medium">Adam/RMSProp LR</label>
              <input 
                type="number" step="0.001" value={params.adamLearningRate || ''} 
                onChange={e => {
                  const val = parseFloat(e.target.value);
                  setParams({...params, adamLearningRate: isNaN(val) ? 0 : val});
                }}
                className="w-full bg-[#F5F5F4] border-none rounded-md px-3 py-2 text-sm focus:ring-1 ring-[#1C1917]"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[11px] font-medium">Epochs</label>
              <input 
                type="number" min="1" max={getDynamicEpochLimit(trainTensors?.y.length || 0)} value={params.epochs || ''} 
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
        
        {/* System Info */}
        <section className="bg-emerald-50/50 border border-emerald-100 p-4 rounded-2xl space-y-2">
          <div className="flex items-center gap-2 text-[10px] font-bold text-emerald-700 uppercase tracking-wider">
            <Zap className="w-3 h-3" />
            Training Environment
          </div>
          <p className="text-[11px] text-emerald-800 leading-relaxed">
            Training is performed <b>locally on your CPU</b> using multi-threaded <b>Web Workers</b>. 
            Your data never leaves your browser, and the app works <b>offline</b> once loaded.
          </p>
          <div className="flex items-center gap-3 pt-1">
            <div className="flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[9px] font-medium text-emerald-700">Local Processing</span>
            </div>
            <div className="flex items-center gap-1">
              <CheckCircle2 className="w-2.5 h-2.5 text-emerald-600" />
              <span className="text-[9px] font-medium text-emerald-700">Privacy Guaranteed</span>
            </div>
          </div>
        </section>

        {/* Datasets Section */}
        <section className="space-y-4">
          <div className="flex items-center justify-between text-xs font-semibold text-[#78716C] uppercase tracking-wider">
            <div className="flex items-center gap-2">
              <Database className="w-3 h-3" />
              Saved Datasets
            </div>
            <button 
              onClick={fetchDatasets}
              className="p-1 hover:bg-[#F5F5F4] rounded-full transition-colors"
              title="Refresh Datasets"
            >
              <RefreshCw className="w-3 h-3" />
            </button>
          </div>
          <div className="space-y-2 max-h-48 overflow-y-auto pr-2 custom-scrollbar">
            {datasets.length === 0 ? (
              <div className="text-[10px] text-[#A8A29E] italic text-center py-4 border border-dashed border-[#E7E5E4] rounded-xl">
                No datasets saved yet.
              </div>
            ) : (
              datasets.map((ds) => (
                <div key={ds.id} className="p-3 bg-white border border-[#E7E5E4] rounded-xl hover:border-[#1C1917] transition-all group">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[11px] font-bold truncate max-w-[120px]" title={ds.name}>{ds.name}</span>
                    <span className={cn(
                      "text-[8px] px-1.5 py-0.5 rounded-full font-bold uppercase",
                      ds.type === 'train' ? "bg-emerald-100 text-emerald-700" : "bg-blue-100 text-blue-700"
                    )}>
                      {ds.type}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[9px] text-[#78716C]">
                    <span>{ds.rowCount.toLocaleString()} rows • {ds.columnCount} cols</span>
                    <div className="flex items-center gap-2">
                      <button 
                        onClick={() => loadDatasetMetadata(ds)}
                        className="text-blue-500 hover:text-blue-700 opacity-0 group-hover:opacity-100 transition-all flex items-center gap-1"
                        title="Load Metadata"
                      >
                        <Download className="w-3 h-3" /> Load
                      </button>
                      <span className="text-[8px] opacity-0 group-hover:opacity-100 transition-opacity">
                        {new Date(ds.timestamp).toLocaleDateString()}
                      </span>
                      <button 
                        onClick={() => deleteDataset(ds.id)}
                        className="text-red-400 hover:text-red-600 opacity-0 group-hover:opacity-100 transition-all"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        <div className="flex flex-col gap-3 mt-auto">
          <div className="flex gap-3">
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
                disabled={isSingleDatasetMode ? !trainFile : (!trainFile || !testFile)}
                className="flex-1 bg-[#1C1917] text-white py-3 rounded-xl font-semibold flex items-center justify-center gap-2 hover:bg-[#44403C] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Play className="w-4 h-4 fill-current" />
                Start
              </button>
            )}
            {isAuthReady && !user && !isTraining && (
              <div className="mt-2 p-3 bg-amber-50 border border-amber-100 rounded-xl flex items-start gap-2 text-amber-800 text-[10px] leading-relaxed">
                <AlertCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                <div>
                  <span className="font-bold">Not Logged In:</span> Your experiment results will not be saved to history. 
                  <button onClick={handleLogin} className="ml-1 underline font-bold hover:text-amber-900">Login with Google</button> to enable history.
                </div>
              </div>
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
          {!isTraining && (trainTensors || testTensors) && (
            <button 
              onClick={clearData}
              className="w-full bg-white border border-[#E7E5E4] text-[#78716C] py-2 rounded-xl text-xs font-medium hover:bg-red-50 hover:text-red-600 hover:border-red-200 transition-all flex items-center justify-center gap-2"
            >
              <Scissors className="w-3 h-3" />
              Clear Memory
            </button>
          )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8 overflow-y-auto space-y-8">
        {/* Error Display */}
        <AnimatePresence>
          {isProcessing && (
            <motion.div 
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-blue-50 border border-blue-200 text-blue-700 px-6 py-4 rounded-2xl flex flex-col gap-2 shadow-sm"
            >
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                <div className="flex-1 text-sm font-medium">{statusMessage}</div>
              </div>
              <div className="w-full bg-blue-100 h-1.5 rounded-full overflow-hidden">
                <motion.div 
                  className="bg-blue-600 h-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${parseProgress}%` }}
                  transition={{ duration: 0.3 }}
                />
              </div>
            </motion.div>
          )}
          {error && (
            <motion.div 
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-red-50 border border-red-200 text-red-700 px-6 py-4 rounded-2xl flex items-center gap-3 shadow-sm"
            >
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <div className="flex-1 text-sm font-medium">{error}</div>
              <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600 transition-colors">
                <Scissors className="w-4 h-4 rotate-90" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Large Dataset Warning */}
        <AnimatePresence>
          {showWarning && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm"
            >
              <div className="bg-white rounded-3xl p-8 max-w-md w-full shadow-2xl border border-[#E7E5E4]">
                <div className="bg-amber-100 w-12 h-12 rounded-2xl flex items-center justify-center mb-6">
                  <AlertCircle className="text-amber-600 w-6 h-6" />
                </div>
                <h3 className="text-xl font-bold mb-2">Large Dataset Detected</h3>
                <p className="text-[#78716C] text-sm leading-relaxed mb-8">
                  You've uploaded a dataset with more than 100,000 samples. Training on this many samples in the browser may be slow or cause performance issues. Consider reducing the sample size for a smoother experience.
                </p>
                <div className="flex gap-3">
                  <button 
                    onClick={() => setShowWarning(false)}
                    className="flex-1 bg-[#1C1917] text-white py-3 rounded-xl font-semibold hover:bg-[#44403C] transition-colors"
                  >
                    Continue Anyway
                  </button>
                  <button 
                    onClick={() => {
                      setShowWarning(false);
                      setTrainSampleSize(50000);
                    }}
                    className="flex-1 bg-[#F5F5F4] text-[#1C1917] py-3 rounded-xl font-semibold hover:bg-[#E7E5E4] transition-colors"
                  >
                    Reduce Size
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Dataset Previews */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
          {trainData.length > 0 && (
            <section className="bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm">
              <div className="flex justify-between items-center mb-4">
                <h2 className="font-bold text-lg">Training Preview</h2>
                <div className="flex flex-col items-end gap-1 text-[10px] font-medium text-[#78716C]">
                  <span>Showing 10 of {trainTensors?.y.length.toLocaleString()} rows</span>
                  <span>Showing 20 of {features.length} columns</span>
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
                <div className="flex flex-col items-end gap-1 text-[10px] font-medium text-[#78716C]">
                  <span>Showing 10 of {testTensors?.y.length.toLocaleString()} rows</span>
                  <span>Showing 20 of {features.length} columns</span>
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
          <section className="bg-[#1C1917] text-white rounded-3xl p-8 shadow-2xl mb-8">
            <div className="flex justify-between items-center mb-8">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-emerald-500 rounded-2xl flex items-center justify-center animate-pulse">
                  <Play className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-black tracking-tight">Parallel Training Active</h2>
                  <p className="text-sm text-[#A8A29E]">Optimizing neural network across multiple cores</p>
                </div>
              </div>
              {isPaused && (
                <span className="px-2 py-1 bg-amber-500/20 text-amber-500 text-[10px] font-bold rounded uppercase tracking-wider animate-pulse">
                  Paused
                </span>
              )}
              <div className="text-right">
                <div className="text-[#A8A29E] text-[10px] font-bold uppercase tracking-widest mb-1">Total Elapsed Time</div>
                <div className="text-2xl font-black">{elapsedTime}s</div>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-6">
              {Object.entries(optimizerProgress).map(([opt, data]: [string, any]) => (
                <div key={opt} className="bg-white/5 border border-white/10 rounded-2xl p-5 space-y-4">
                  <div className="flex justify-between items-center">
                    <div className="font-bold text-sm tracking-tight">{opt}</div>
                    <div className="text-[10px] font-bold text-[#A8A29E] uppercase">Epoch {data.epoch}/{params.epochs}</div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="space-y-1">
                      <div className="flex justify-between text-[9px] font-bold text-[#A8A29E] uppercase">
                        <span>Training</span>
                        <span>{Math.round(data.trainProgress)}%</span>
                      </div>
                      <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                        <motion.div 
                          className="h-full bg-emerald-500"
                          initial={{ width: 0 }}
                          animate={{ width: `${data.trainProgress}%` }}
                        />
                      </div>
                    </div>

                    <div className="space-y-1">
                      <div className="flex justify-between text-[9px] font-bold text-[#A8A29E] uppercase">
                        <span>Testing</span>
                        <span>{Math.round(data.testProgress)}%</span>
                      </div>
                      <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                        <motion.div 
                          className="h-full bg-blue-500"
                          initial={{ width: 0 }}
                          animate={{ width: `${data.testProgress}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Aggregated Progress */}
            <div className="mt-8 pt-8 border-t border-white/10">
              <div className="flex justify-between text-[10px] font-bold text-[#A8A29E] uppercase tracking-widest mb-2">
                <span>Overall Completion</span>
                <span>{Math.round(overallProgress)}%</span>
              </div>
              <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                <motion.div 
                  className="h-full bg-gradient-to-r from-emerald-500 to-blue-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${overallProgress}%` }}
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
                <BarChart3 className="w-4 h-4" /> 
                Loss vs Epoch
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F5F5F4" />
                    <XAxis 
                      dataKey="epoch" 
                      axisLine={true} 
                      tickLine={true} 
                      tick={{fontSize: 10}} 
                      label={{ value: 'Epoch', position: 'insideBottom', offset: -15, fontSize: 12, fontWeight: 'bold' }}
                    />
                    <YAxis 
                      axisLine={true} 
                      tickLine={true} 
                      tick={{fontSize: 10}} 
                      label={{ value: 'Loss', angle: -90, position: 'insideLeft', offset: 10, fontSize: 12, fontWeight: 'bold' }}
                    />
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
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F5F5F4" />
                    <XAxis 
                      dataKey="epoch" 
                      axisLine={true} 
                      tickLine={true} 
                      tick={{fontSize: 10}} 
                      label={{ value: 'Epoch', position: 'insideBottom', offset: -15, fontSize: 12, fontWeight: 'bold' }}
                    />
                    <YAxis 
                      axisLine={true} 
                      tickLine={true} 
                      tick={{fontSize: 10}} 
                      domain={[0, 1]} 
                      label={{ value: 'Accuracy', angle: -90, position: 'insideLeft', offset: 10, fontSize: 12, fontWeight: 'bold' }}
                    />
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

            {/* New Train vs Test Accuracy Chart */}
            <section className="bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm col-span-2">
              <h3 className="font-bold mb-6 flex items-center gap-2">
                <Activity className="w-4 h-4" /> 
                Train vs Test Accuracy Comparison (All Optimizers)
              </h3>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F5F5F4" />
                    <XAxis 
                      dataKey="epoch" 
                      axisLine={true} 
                      tickLine={true} 
                      tick={{fontSize: 10}} 
                      label={{ value: 'Epoch', position: 'insideBottom', offset: -15, fontSize: 12, fontWeight: 'bold' }}
                    />
                    <YAxis 
                      axisLine={true} 
                      tickLine={true} 
                      tick={{fontSize: 10}} 
                      domain={[0, 1]} 
                      label={{ value: 'Accuracy', angle: -90, position: 'insideLeft', offset: 10, fontSize: 12, fontWeight: 'bold' }}
                    />
                    <Tooltip contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)'}} />
                    <Legend iconType="circle" wrapperStyle={{fontSize: 10, paddingTop: 20}} />
                    {results.map((res, i) => {
                      const colors = ['#1C1917', '#D97706', '#059669', '#2563EB'];
                      return (
                        <React.Fragment key={res.optimizer}>
                          <Line 
                            type="monotone" 
                            dataKey={`${res.optimizer}_train_acc`} 
                            stroke={colors[i]} 
                            strokeWidth={2} 
                            strokeDasharray="5 5"
                            dot={false}
                            name={`${res.optimizer} (Train)`}
                          />
                          <Line 
                            type="monotone" 
                            dataKey={`${res.optimizer}_test_acc`} 
                            stroke={colors[i]} 
                            strokeWidth={2} 
                            dot={false}
                            name={`${res.optimizer} (Test)`}
                          />
                        </React.Fragment>
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 flex gap-4 text-[10px] font-medium text-[#78716C] justify-center">
                <div className="flex items-center gap-1">
                  <div className="w-3 h-0.5 bg-[#78716C] border-t border-dashed" />
                  <span>Dashed: Training Accuracy</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-0.5 bg-[#78716C]" />
                  <span>Solid: Testing Accuracy</span>
                </div>
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
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F5F5F4" />
                    <XAxis 
                      dataKey="epoch" 
                      axisLine={true} 
                      tickLine={true} 
                      tick={{fontSize: 10}} 
                      label={{ value: 'Epoch', position: 'insideBottom', offset: -15, fontSize: 12, fontWeight: 'bold' }}
                    />
                    <YAxis 
                      axisLine={true} 
                      tickLine={true} 
                      tick={{fontSize: 10}} 
                      label={{ value: 'Grad Norm', angle: -90, position: 'insideLeft', offset: 10, fontSize: 12, fontWeight: 'bold' }}
                    />
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
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F5F5F4" />
                    <XAxis 
                      dataKey="epoch" 
                      axisLine={true} 
                      tickLine={true} 
                      tick={{fontSize: 10}} 
                      label={{ value: 'Epoch', position: 'insideBottom', offset: -15, fontSize: 12, fontWeight: 'bold' }}
                    />
                    <YAxis 
                      axisLine={true} 
                      tickLine={true} 
                      tick={{fontSize: 10}} 
                      label={{ value: 'Update Ratio', angle: -90, position: 'insideLeft', offset: 10, fontSize: 12, fontWeight: 'bold' }}
                    />
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
            <div className="flex justify-between items-center mb-6">
              <h2 className="font-bold text-lg">Optimizer Comparison</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="bg-[#F5F5F4] text-[#78716C] uppercase text-[10px] tracking-wider">
                  <tr>
                    <th className="px-6 py-4 font-semibold">Optimizer</th>
                    <th className="px-6 py-4 font-semibold">Accuracy</th>
                    <th className="px-6 py-4 font-semibold">F1 Score</th>
                    <th className="px-6 py-4 font-semibold">Log Loss</th>
                    <th className="px-6 py-4 font-semibold">Conv. Epoch (85%)</th>
                    <th className="px-6 py-4 font-semibold">Avg Grad Norm</th>
                    <th className="px-6 py-4 font-semibold">Avg Update Ratio</th>
                    <th className="px-6 py-4 font-semibold">Throughput</th>
                    <th className="px-6 py-4 font-semibold">Train Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E7E5E4]">
                  {results.map((res) => (
                    <tr key={res.optimizer} className={cn(bestOptimizer?.optimizer === res.optimizer && "bg-emerald-50/50")}>
                      <td className="px-6 py-4 font-bold flex items-center gap-2">
                        {res.optimizer}
                        {bestOptimizer?.optimizer === res.optimizer && <CheckCircle2 className="w-3 h-3 text-emerald-600" />}
                      </td>
                      <td className="px-6 py-4">{safeFixed(res.testAccuracy, 1, 100, '%')}</td>
                      <td className="px-6 py-4">{safeFixed(res.f1Score, 1, 100, '%')}</td>
                      <td className="px-6 py-4">{safeFixed(res.logLoss, 4)}</td>
                      <td className="px-6 py-4">{res.convergenceEpoch85 || 'N/A'}</td>
                      <td className="px-6 py-4">{safeFixed(res.avgGradientNorm, 4)}</td>
                      <td className="px-6 py-4">{safeFixed(res.avgUpdateRatio, 6)}</td>
                      <td className="px-6 py-4">{safeFixed(res.avgThroughput, 0, 1, ' img/s')}</td>
                      <td className="px-6 py-4">{safeFixed(res.trainingTime, 1, 1, 's')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Metric Definitions Info Box */}
        {results.length > 0 && (
          <section className="bg-blue-50 border border-blue-100 p-6 rounded-2xl">
            <h3 className="font-bold text-blue-900 mb-4 flex items-center gap-2">
              <Info className="w-4 h-4" /> Understanding the Metrics
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="space-y-1">
                <div className="text-[11px] font-bold text-blue-800 uppercase tracking-wider">Accuracy</div>
                <p className="text-xs text-blue-700 leading-relaxed">
                  The percentage of total correct predictions. Indicates the model's overall correctness.
                </p>
              </div>
              <div className="space-y-1">
                <div className="text-[11px] font-bold text-blue-800 uppercase tracking-wider">Log Loss</div>
                <p className="text-xs text-blue-700 leading-relaxed">
                  Measures the performance of a classification model where the prediction input is a <b>probability value</b>. Lower is better.
                </p>
              </div>
              <div className="space-y-1">
                <div className="text-[11px] font-bold text-blue-800 uppercase tracking-wider">Gradient Norm</div>
                <p className="text-xs text-blue-700 leading-relaxed">
                  Magnitude of gradients. High values indicate <b>exploding gradients</b>, while near-zero values indicate <b>vanishing gradients</b> or convergence.
                </p>
              </div>
              <div className="space-y-1">
                <div className="text-[11px] font-bold text-blue-800 uppercase tracking-wider">Update Ratio</div>
                <p className="text-xs text-blue-700 leading-relaxed">
                  Ratio of update magnitude to parameter magnitude. A healthy ratio is around <b>1e-3</b>. Too high means learning rate is too large.
                </p>
              </div>
              <div className="space-y-1">
                <div className="text-[11px] font-bold text-blue-800 uppercase tracking-wider">Convergence Epoch (85%)</div>
                <p className="text-xs text-blue-700 leading-relaxed">
                  The first epoch where the model reaches <b>85% test accuracy</b>. Measures the speed of optimization.
                </p>
              </div>
              <div className="space-y-1">
                <div className="text-[11px] font-bold text-blue-800 uppercase tracking-wider">Throughput</div>
                <p className="text-xs text-blue-700 leading-relaxed">
                  Number of training samples processed per second. Measures the <b>computational efficiency</b> of the optimizer.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* Analysis Section */}
        {bestOptimizer && (
          <div className="space-y-8">
            <div className="grid grid-cols-3 gap-8">
              <section className="col-span-1 bg-emerald-600 text-white rounded-2xl p-6 shadow-lg flex flex-col justify-between">
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-widest opacity-80 mb-2">Best Optimizer</h3>
                  <div className="text-3xl font-black mb-4">{bestOptimizer.optimizer}</div>
                  <p className="text-sm leading-relaxed opacity-90">
                    The best optimizer for this dataset is <span className="font-bold">{bestOptimizer.optimizer}</span>, selected for its superior balanced performance across accuracy, stability, and speed.
                  </p>
                </div>
                <div className="mt-6 pt-6 border-t border-white/20">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-[10px] uppercase font-bold opacity-60">Test Accuracy</div>
                      <div className="text-xl font-black">{safeFixed(bestOptimizer.testAccuracy, 2, 100, '%')}</div>
                    </div>
                    <div>
                      <div className="text-[10px] uppercase font-bold opacity-60">F1 Score</div>
                      <div className="text-xl font-black">{safeFixed(bestOptimizer.f1Score, 2, 100, '%')}</div>
                    </div>
                    <div>
                      <div className="text-[10px] uppercase font-bold opacity-60">Log Loss</div>
                      <div className="text-xl font-black">{safeFixed(bestOptimizer.logLoss, 4)}</div>
                    </div>
                    <div>
                      <div className="text-[10px] uppercase font-bold opacity-60">Conv. Epoch (85%)</div>
                      <div className="text-xl font-black">{bestOptimizer.convergenceEpoch85 || 'N/A'}</div>
                    </div>
                  </div>
                </div>
              </section>

              <section className="col-span-2 bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm">
                <h3 className="font-bold mb-4 flex items-center gap-2">
                  <Info className="w-4 h-4 text-[#78716C]" /> Optimizer Performance Analysis
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  {results.map(res => (
                    <div key={res.optimizer} className={cn("p-3 rounded-xl transition-all", bestOptimizer.optimizer === res.optimizer ? "bg-emerald-50 border border-emerald-100" : "bg-[#F5F5F4]")}>
                      <div className="flex justify-between items-center mb-1">
                        <div className="text-xs font-bold">{res.optimizer}</div>
                        {bestOptimizer.optimizer === res.optimizer && <span className="text-[8px] bg-emerald-600 text-white px-1.5 py-0.5 rounded-full font-bold uppercase tracking-tighter">Winner</span>}
                      </div>
                      <p className="text-[11px] text-[#78716C] leading-tight">
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

          </div>
        )}

        {/* History Dashboard */}
        <section className="bg-white rounded-2xl p-6 border border-[#E7E5E4] shadow-sm">
          <div className="flex justify-between items-center mb-6">
            <h2 className="font-bold text-lg flex items-center gap-2">
              <History className="w-5 h-5" /> Experiment History
            </h2>
            <button 
              onClick={() => fetchHistory()}
              className="p-2 hover:bg-[#F5F5F4] rounded-lg transition-colors text-[#78716C]"
              title="Refresh History"
            >
              <Activity className="w-4 h-4" />
            </button>
          </div>
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
                      </div>
                      <div className="text-sm font-bold">{safeFixed(exp.test_accuracy, 1, 100, '%')}</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[#78716C] uppercase font-semibold flex items-center justify-end gap-1">
                        Time
                      </div>
                      <div className="text-sm font-bold">{safeFixed(exp.execution_time, 1, 1, 's')}</div>
                    </div>
                    <div className="pl-4 border-l border-[#E7E5E4] flex items-center gap-2">
                      <button 
                        onClick={(e) => deleteExperiment(exp.id, e)}
                        className={cn(
                          "p-2 rounded-full transition-all",
                          confirmDeleteId === exp.id 
                            ? "bg-red-600 text-white opacity-100" 
                            : "bg-white text-red-600 opacity-0 group-hover:opacity-100 hover:bg-red-50"
                        )}
                        title={confirmDeleteId === exp.id ? "Click again to confirm" : "Delete Experiment"}
                      >
                        {confirmDeleteId === exp.id ? <Check className="w-4 h-4" /> : <Trash2 className="w-4 h-4" />}
                      </button>
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
                    The model converged with a rate of <span className="font-bold">{safeFixed(selectedExperiment.convergence_rate, 4)}</span> over <span className="font-bold">{safeFixed(selectedExperiment.execution_time, 2, 1, 's')}</span>.
                    {selectedExperiment.optimizer === 'Adam' ? " Adam's adaptive learning rate helped in stable convergence." : ""}
                    {selectedExperiment.test_accuracy > 0.8 ? " The high accuracy suggests well-tuned parameters for this specific data." : " There might be room for improvement by adjusting the learning rate or hidden layer size."}
                  </p>
                </section>

                {/* Graphs */}
                <div className="grid grid-cols-2 gap-8">
                  <div className="bg-white border border-[#E7E5E4] p-6 rounded-2xl" id="report-chart-loss">
                    <h4 className="font-bold mb-4 text-sm flex items-center gap-1">
                      Loss & Accuracy
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

                  <div className="bg-white border border-[#E7E5E4] p-6 rounded-2xl col-span-2" id="report-chart-train-test">
                    <h4 className="font-bold mb-4 text-sm flex items-center gap-1">
                      Train vs Test Accuracy
                    </h4>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={selectedExperiment.logs}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F5F5F4" />
                          <XAxis dataKey="epoch" tick={{fontSize: 10}} />
                          <YAxis tick={{fontSize: 10}} domain={[0, 1]} />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="trainAccuracy" stroke="#059669" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Train Accuracy" />
                          <Line type="monotone" dataKey="testAccuracy" stroke="#2563EB" strokeWidth={2} dot={false} name="Test Accuracy" />
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
                    </div>
                    <div className="text-xl font-black text-blue-600">{safeFixed(selectedExperiment.f1_score, 2, 100, '%')}</div>
                  </div>
                  <div className="p-4 bg-white border border-[#E7E5E4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1 flex items-center gap-1">
                      Log Loss
                    </div>
                    <div className="text-xl font-black text-rose-600">{safeFixed(selectedExperiment.log_loss, 4)}</div>
                  </div>
                  <div className="p-4 bg-white border border-[#E7E5E4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1 flex items-center gap-1">
                      AULC
                    </div>
                    <div className="text-xl font-black text-amber-600">{safeFixed(selectedExperiment.aulc, 2)}</div>
                  </div>
                  <div className="p-4 bg-white border border-[#E7E5E4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1 flex items-center gap-1">
                      Loss Variance
                    </div>
                    <div className="text-xl font-black">{safeFixed(selectedExperiment.loss_variance, 6)}</div>
                  </div>
                  <div className="p-4 bg-white border border-[#E7E5E4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1 flex items-center gap-1">
                      Conv. Rate
                    </div>
                    <div className="text-xl font-black text-emerald-600">{safeFixed(selectedExperiment.convergence_rate, 4)}</div>
                  </div>
                  <div className="p-4 bg-white border border-[#E7E5E4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1 flex items-center gap-1">
                      Throughput
                    </div>
                    <div className="text-xl font-black text-indigo-600">{safeFixed(selectedExperiment.logs[selectedExperiment.logs.length - 1].throughput, 1)} <span className="text-[10px] font-normal text-[#78716C]">s/s</span></div>
                  </div>
                  <div className="p-4 bg-white border border-[#E7E5E4] rounded-2xl">
                    <div className="text-[10px] text-[#78716C] uppercase font-bold mb-1 flex items-center gap-1">
                      Param Norm
                    </div>
                    <div className="text-xl font-black text-slate-600">{safeFixed(selectedExperiment.logs[selectedExperiment.logs.length - 1].parameterNorm, 2)}</div>
                  </div>
                </div>

                {/* ANOVA Table (Only in Report Section) */}
                {results.length >= 2 && (
                  <div className="bg-white border border-[#E7E5E4] p-6 rounded-2xl">
                    <h4 className="font-bold mb-4 text-sm flex items-center gap-2">
                      <BarChart3 className="w-4 h-4" /> 
                      Statistical Comparison (ANOVA)
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left">
                        <thead className="bg-[#F5F5F4] text-[#78716C] uppercase text-[10px] tracking-wider">
                          <tr>
                            <th className="px-4 py-2">Source</th>
                            <th className="px-4 py-2">DF</th>
                            <th className="px-4 py-2">SS</th>
                            <th className="px-4 py-2">MS</th>
                            <th className="px-4 py-2">F</th>
                            <th className="px-4 py-2">P-Value</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#E7E5E4]">
                          {anovaResult && (
                            <>
                              <tr>
                                <td className="px-4 py-2 font-bold">Between Groups</td>
                                <td className="px-4 py-2">{anovaResult.dfBetween}</td>
                                <td className="px-4 py-2">{safeFixed(anovaResult.ssBetween, 6)}</td>
                                <td className="px-4 py-2">{safeFixed(anovaResult.msBetween, 6)}</td>
                                <td className="px-4 py-2 font-bold">{safeFixed(anovaResult.fValue, 4)}</td>
                                <td className="px-4 py-2">{anovaResult.pValue < 0.05 ? '< 0.05' : '> 0.05'}</td>
                              </tr>
                              <tr>
                                <td className="px-4 py-2 font-bold">Within Groups</td>
                                <td className="px-4 py-2">{anovaResult.dfWithin}</td>
                                <td className="px-4 py-2">{safeFixed(anovaResult.ssWithin, 6)}</td>
                                <td className="px-4 py-2">{safeFixed(anovaResult.msWithin, 6)}</td>
                                <td className="px-4 py-2"></td>
                                <td className="px-4 py-2"></td>
                              </tr>
                            </>
                          )}
                        </tbody>
                      </table>
                    </div>
                    <p className="mt-4 text-[10px] text-[#78716C] italic">
                      * ANOVA performed on the accuracy of the final 5 epochs across all optimizers.
                    </p>
                  </div>
                )}

                {/* Advanced Visualizations */}
                <div className="grid grid-cols-2 gap-8">
                  <div className="bg-white border border-[#E7E5E4] p-6 rounded-2xl" id="report-chart-advanced-1">
                    <h4 className="font-bold mb-4 text-sm flex items-center gap-2">
                      <Activity className="w-4 h-4" /> 
                      Gradient & Loss Variance
                    </h4>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={selectedExperiment.logs}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F5F5F4" />
                          <XAxis dataKey="epoch" tick={{fontSize: 10}} />
                          <YAxis yAxisId="left" tick={{fontSize: 10}} />
                          <YAxis yAxisId="right" orientation="right" tick={{fontSize: 10}} />
                          <Tooltip />
                          <Line yAxisId="left" type="monotone" dataKey="gradientVariance" stroke="#F43F5E" strokeWidth={2} dot={false} name="Grad Var" />
                          <Line yAxisId="right" type="monotone" dataKey="lossVariance" stroke="#10B981" strokeWidth={2} dot={false} name="Loss Var" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div className="bg-white border border-[#E7E5E4] p-6 rounded-2xl" id="report-chart-advanced-2">
                    <h4 className="font-bold mb-4 text-sm flex items-center gap-2">
                      <Zap className="w-4 h-4" /> 
                      Parameter Norm & Throughput
                    </h4>
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={selectedExperiment.logs}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F5F5F4" />
                          <XAxis dataKey="epoch" tick={{fontSize: 10}} />
                          <YAxis yAxisId="left" tick={{fontSize: 10}} />
                          <YAxis yAxisId="right" orientation="right" tick={{fontSize: 10}} />
                          <Tooltip />
                          <Line yAxisId="left" type="monotone" dataKey="parameterNorm" stroke="#3B82F6" strokeWidth={2} dot={false} name="Param Norm" />
                          <Line yAxisId="right" type="monotone" dataKey="throughput" stroke="#F59E0B" strokeWidth={2} dot={false} name="Throughput" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-8">
                  <div className="bg-white border border-[#E7E5E4] p-6 rounded-2xl" id="report-chart-speed">
                    <h4 className="font-bold mb-4 text-sm flex items-center gap-2">
                      <TrendingDown className="w-4 h-4" /> 
                      Convergence Speed
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
                          <th className="px-6 py-3 font-semibold">Loss Var</th>
                          <th className="px-6 py-3 font-semibold">Grad Var</th>
                          <th className="px-6 py-3 font-semibold">Param Norm</th>
                          <th className="px-6 py-3 font-semibold">Throughput</th>
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
                            <td className="px-6 py-3">{safeFixed(m.lossVariance, 6)}</td>
                            <td className="px-6 py-3">{safeFixed(m.gradientVariance, 6)}</td>
                            <td className="px-6 py-3">{safeFixed(m.parameterNorm, 4)}</td>
                            <td className="px-6 py-3">{safeFixed(m.throughput, 1)}</td>
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
                    </div>
                    <div className="text-2xl font-black">{safeFixed(selectedExperiment.precision, 1, 100, '%')}</div>
                  </div>
                  <div className="p-6 border border-[#E7E5E4] rounded-2xl text-center">
                    <div className="text-xs text-[#78716C] uppercase font-bold mb-1 flex items-center justify-center gap-1">
                      Recall
                    </div>
                    <div className="text-2xl font-black">{safeFixed(selectedExperiment.recall, 1, 100, '%')}</div>
                  </div>
                  <div className="p-6 border border-[#E7E5E4] rounded-2xl text-center">
                    <div className="text-xs text-[#78716C] uppercase font-bold mb-1 flex items-center justify-center gap-1">
                      F1 Score
                    </div>
                    <div className="text-2xl font-black">{safeFixed(selectedExperiment.f1_score, 1, 100, '%')}</div>
                  </div>
                  <div className="p-6 border border-[#E7E5E4] rounded-2xl text-center">
                    <div className="text-xs text-[#78716C] uppercase font-bold mb-1 flex items-center justify-center gap-1">
                      Log Loss
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

      <InfoModal isOpen={isHelpOpen} onClose={() => setIsHelpOpen(false)} />
    </div>
  );
}
