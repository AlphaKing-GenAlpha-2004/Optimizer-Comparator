import express from 'express';
import { createServer as createViteServer } from 'vite';
import Database from 'better-sqlite3';
import path from 'path';
import multer from 'multer';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const db = new Database(path.join(__dirname, 'experiments.db'));

// Initialize Database
db.exec(`
  CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT,
    sample_size INTEGER,
    train_test_split REAL,
    optimizer TEXT,
    hidden_size INTEGER,
    learning_rate REAL,
    epochs INTEGER,
    batch_size INTEGER,
    test_accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    log_loss REAL,
    convergence_rate REAL,
    training_time REAL,
    testing_time REAL,
    execution_time REAL,
    aulc REAL,
    loss_variance REAL,
    logs TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    type TEXT, -- 'train' or 'test'
    rowCount INTEGER,
    columnCount INTEGER,
    features TEXT, -- JSON array
    target TEXT,
    content TEXT, -- Store small datasets or metadata
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
  );
`);

// Migration: Add missing columns if they don't exist
const tableInfo = db.prepare("PRAGMA table_info(experiments)").all() as any[];
const columnNames = tableInfo.map(c => c.name);

if (!columnNames.includes('training_time')) {
  db.exec("ALTER TABLE experiments ADD COLUMN training_time REAL");
}
if (!columnNames.includes('testing_time')) {
  db.exec("ALTER TABLE experiments ADD COLUMN testing_time REAL");
}
if (!columnNames.includes('aulc')) {
  db.exec("ALTER TABLE experiments ADD COLUMN aulc REAL");
}
if (!columnNames.includes('loss_variance')) {
  db.exec("ALTER TABLE experiments ADD COLUMN loss_variance REAL");
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '50mb' }));

  // Health Check
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
  });

  // API Routes
  app.get('/api/experiments', (req, res) => {
    console.log('GET /api/experiments');
    try {
      const rows = db.prepare('SELECT id, dataset_name, optimizer, test_accuracy, execution_time, timestamp FROM experiments ORDER BY timestamp DESC LIMIT 50').all();
      res.json(rows);
    } catch (error) {
      console.error('Error in GET /api/experiments:', error);
      res.status(500).json({ error: 'Failed to fetch history' });
    }
  });

  app.get('/api/experiments/:id', (req, res) => {
    console.log(`GET /api/experiments/${req.params.id}`);
    try {
      const row = db.prepare('SELECT * FROM experiments WHERE id = ?').get(req.params.id);
      if (!row) {
        return res.status(404).json({ error: 'Experiment not found' });
      }
      res.json(row);
    } catch (error) {
      console.error(`Error in GET /api/experiments/${req.params.id}:`, error);
      res.status(500).json({ error: 'Failed to fetch experiment' });
    }
  });

  app.post('/api/experiments', (req, res) => {
    console.log('POST /api/experiments');
    const { 
      dataset_name, sample_size, train_test_split, optimizer, 
      hidden_size, learning_rate, epochs, batch_size,
      test_accuracy, precision, recall, f1_score,
      log_loss, convergence_rate, training_time, testing_time, execution_time, aulc, loss_variance, logs 
    } = req.body;
    try {
      const stmt = db.prepare(`
        INSERT INTO experiments (
          dataset_name, sample_size, train_test_split, optimizer, 
          hidden_size, learning_rate, epochs, batch_size,
          test_accuracy, precision, recall, f1_score,
          log_loss, convergence_rate, training_time, testing_time, execution_time, aulc, loss_variance, logs
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);
      stmt.run(
        dataset_name, sample_size, train_test_split, optimizer, 
        hidden_size, learning_rate, epochs, batch_size,
        test_accuracy, precision, recall, f1_score, 
        log_loss, convergence_rate, training_time, testing_time, execution_time, aulc, loss_variance,
        typeof logs === 'string' ? logs : JSON.stringify(logs)
      );
      console.log('Successfully saved experiment');
      res.json({ success: true });
    } catch (error) {
      console.error('Error in POST /api/experiments:', error);
      res.status(500).json({ error: 'Failed to save experiment' });
    }
  });

  app.delete('/api/experiments/:id', (req, res) => {
    console.log(`DELETE /api/experiments/${req.params.id}`);
    try {
      db.prepare('DELETE FROM experiments WHERE id = ?').run(req.params.id);
      res.json({ success: true });
    } catch (error) {
      console.error(`Error in DELETE /api/experiments/${req.params.id}:`, error);
      res.status(500).json({ error: 'Failed to delete experiment' });
    }
  });

  app.get('/api/datasets', (req, res) => {
    try {
      const rows = db.prepare('SELECT * FROM datasets ORDER BY timestamp DESC').all();
      res.json(rows);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch datasets' });
    }
  });

  app.post('/api/datasets', (req, res) => {
    const { name, type, rowCount, columnCount, features, target, content } = req.body;
    try {
      const stmt = db.prepare(`
        INSERT INTO datasets (name, type, rowCount, columnCount, features, target, content)
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `);
      stmt.run(name, type, rowCount, columnCount, JSON.stringify(features), target, content);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: 'Failed to save dataset' });
    }
  });

  app.delete('/api/datasets/:id', (req, res) => {
    try {
      db.prepare('DELETE FROM datasets WHERE id = ?').run(req.params.id);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: 'Failed to delete dataset' });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
