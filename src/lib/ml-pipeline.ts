import * as math from 'mathjs';

export type TaskType = 'binary' | 'multiclass' | 'regression';

export interface DataQualityReport {
  missingValues: Record<string, number>;
  placeholders: Record<string, number[]>;
  distributions: Record<string, { mean: number; std: number; skew: number; type: 'continuous' | 'categorical' }>;
  classDistribution: Record<string, number>;
  imbalanceRatio: number;
  suggestedTask: TaskType;
}

export interface PreprocessingConfig {
  imputeStrategy: 'median' | 'mean' | 'most_frequent';
  scalingStrategy: 'standard' | 'minmax' | 'none';
  handleImbalance: boolean;
  addMissingIndicators: boolean;
}

export class DataPipeline {
  private report: DataQualityReport = {
    missingValues: {},
    placeholders: {},
    distributions: {},
    classDistribution: {},
    imbalanceRatio: 1,
    suggestedTask: 'multiclass'
  };

  private featureStats: Record<string, { mean: number; std: number; median: number; mode: any }> = {};

  analyze(data: any[], target: string): DataQualityReport {
    if (data.length === 0) return this.report;

    const columns = Object.keys(data[0]);
    const rowCount = data.length;

    columns.forEach(col => {
      if (col === target) {
        // Analyze target for task detection and imbalance
        const counts: Record<string, number> = {};
        data.forEach(row => {
          const val = String(row[col]);
          counts[val] = (counts[val] || 0) + 1;
        });
        this.report.classDistribution = counts;
        
        const values = Object.values(counts);
        if (values.length === 2) {
          this.report.suggestedTask = 'binary';
        } else if (values.length > 20 && !isNaN(Number(Object.keys(counts)[0]))) {
          this.report.suggestedTask = 'regression';
        } else {
          this.report.suggestedTask = 'multiclass';
        }

        if (values.length >= 2) {
          this.report.imbalanceRatio = Math.max(...values) / Math.min(...values);
        }
        return;
      }

      // Analyze features
      let missingCount = 0;
      let placeholderCount = 0;
      const values: number[] = [];
      const rawValues: any[] = [];

      data.forEach(row => {
        const val = row[col];
        rawValues.push(val);
        if (val === null || val === undefined || val === '' || (typeof val === 'number' && isNaN(val))) {
          missingCount++;
        } else {
          const num = Number(val);
          if (!isNaN(num)) {
            values.push(num);
            // Placeholder detection (e.g., 0 in continuous, -1, 999)
            if (num === 0 || num === -1 || num === 999) {
              placeholderCount++;
            }
          }
        }
      });

      this.report.missingValues[col] = (missingCount / rowCount) * 100;
      
      if (values.length > 0) {
        const mean = math.mean(values) as any as number;
        const std = math.std(values) as any as number;
        const median = math.median(values) as any as number;
        
        // Skewness calculation
        const m3 = values.reduce((acc, v) => acc + Math.pow(v - mean, 3), 0) / values.length;
        const skew = std !== 0 ? m3 / Math.pow(std, 3) : 0;

        const uniqueCount = new Set(values).size;
        const type = uniqueCount < 10 ? 'categorical' : 'continuous';

        this.report.distributions[col] = { mean, std, skew, type };
        this.featureStats[col] = { mean, std, median, mode: this.getMode(rawValues) };
      }
    });

    return this.report;
  }

  private getMode(arr: any[]) {
    const counts: Record<any, number> = {};
    arr.forEach(v => counts[v] = (counts[v] || 0) + 1);
    let maxCount = 0;
    let mode = arr[0];
    for (const k in counts) {
      if (counts[k] > maxCount) {
        maxCount = counts[k];
        mode = k;
      }
    }
    return mode;
  }

  preprocessRow(row: any, target: string, config: PreprocessingConfig): { x: number[]; y: any } {
    const columns = Object.keys(row).filter(c => c !== target);
    const x: number[] = [];
    const y = row[target];

    columns.forEach(col => {
      let val = row[col];
      const isMissing = val === null || val === undefined || val === '' || (typeof val === 'number' && isNaN(val));
      const isPlaceholder = (val === 0 && this.report.distributions[col]?.type === 'continuous' && (this.report.missingValues[col] || 0) > 20);

      if (isMissing || isPlaceholder) {
        const stats = this.featureStats[col];
        if (config.imputeStrategy === 'median') {
          val = stats?.median ?? 0;
        } else if (config.imputeStrategy === 'most_frequent') {
          val = stats?.mode ?? 0;
        } else {
          val = stats?.mean ?? 0;
        }
        if (config.addMissingIndicators) {
          x.push(1);
        }
      } else {
        if (config.addMissingIndicators) {
          x.push(0);
        }
      }

      val = Number(val);
      if (isNaN(val)) val = 0;

      const dist = this.report.distributions[col];
      const stats = this.featureStats[col];

      // Log transform for skewed continuous data
      if (dist && dist.type === 'continuous' && Math.abs(dist.skew) > 1 && val > 0) {
        val = Math.log1p(val);
      }

      // Scaling
      if (config.scalingStrategy === 'standard' && stats && stats.std !== 0) {
        val = (val - stats.mean) / stats.std;
      } else if (config.scalingStrategy === 'minmax') {
        val = (val - stats.mean) / (stats.std * 4 + 1e-7); 
      }

      x.push(val);
    });

    return { x, y };
  }
}
