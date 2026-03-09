import express from 'express';
import { createServer as createViteServer } from 'vite';
import Database from 'better-sqlite3';
import path from 'path';
import multer from 'multer';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const db = new Database('experiments.db');

// Initialize Database
db.exec(`
  CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT,
    sample_size INTEGER,
    optimizer TEXT,
    test_accuracy REAL,
    convergence_rate REAL,
    execution_time REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '50mb' }));

  // API Routes
  app.get('/api/history', (req, res) => {
    try {
      const rows = db.prepare('SELECT * FROM experiments ORDER BY timestamp DESC LIMIT 50').all();
      res.json(rows);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch history' });
    }
  });

  app.post('/api/experiments', (req, res) => {
    const { dataset_name, sample_size, optimizer, test_accuracy, convergence_rate, execution_time } = req.body;
    try {
      const stmt = db.prepare(`
        INSERT INTO experiments (dataset_name, sample_size, optimizer, test_accuracy, convergence_rate, execution_time)
        VALUES (?, ?, ?, ?, ?, ?)
      `);
      stmt.run(dataset_name, sample_size, optimizer, test_accuracy, convergence_rate, execution_time);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: 'Failed to save experiment' });
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
