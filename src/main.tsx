import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { ensureCrossOriginIsolation } from './lib/coi';
import './index.css';

// Before anything else: without cross-origin isolation onnxruntime-web is stuck on one
// thread, which costs roughly 5x on the deployed site.
ensureCrossOriginIsolation();

const container = document.getElementById('root');
if (!container) throw new Error('#root is missing from index.html');

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
