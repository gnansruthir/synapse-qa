// Main frontend logic for SynapseQA
let nodes = [];
const nodeCount = 28;
let angleY = 0.002; // Small base rotation speed
let mouseX = 0;
let mouseY = 0;

document.addEventListener('DOMContentLoaded', () => {
  initStarfield();
  init3DGraph();
  initQueryConsole();
});

// 1. Drifting background starfield
function initStarfield() {
  const canvas = document.getElementById('star-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  let w = (canvas.width = window.innerWidth);
  let h = (canvas.height = window.innerHeight);
  
  const stars = [];
  const count = 70;
  
  for (let i = 0; i < count; i++) {
    stars.push({
      x: Math.random() * w,
      y: Math.random() * h,
      size: Math.random() * 1.5 + 0.5,
      speed: Math.random() * 0.1 + 0.05
    });
  }
  
  function animate() {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(232, 224, 255, 0.4)';
    stars.forEach(s => {
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
      ctx.fill();
      
      // Drift left
      s.x -= s.speed;
      if (s.x < 0) s.x = w;
    });
    requestAnimationFrame(animate);
  }
  
  window.addEventListener('resize', () => {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  });
  
  animate();
}

// 2. Rotating 3D Knowledge Graph Constellation
function init3DGraph() {
  const canvas = document.getElementById('graph-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  const size = 450;
  canvas.width = size;
  canvas.height = size;
  
  // Track mouse coordinates for parallax
  document.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX - window.innerWidth / 2) / 100;
    mouseY = (e.clientY - window.innerHeight / 2) / 100;
  });
  
  // Initialize 3D nodes
  for (let i = 0; i < nodeCount; i++) {
    // Random spherical coords
    const u = Math.random();
    const v = Math.random();
    const theta = u * 2.0 * Math.PI;
    const phi = Math.acos(2.0 * v - 1.0);
    const r = 160; // sphere radius
    
    // Convert to 3D Cartesian coords
    const x = r * Math.sin(phi) * Math.cos(theta);
    const y = r * Math.sin(phi) * Math.sin(theta);
    const z = r * Math.cos(phi);
    
    // Node properties
    let color = 'rgba(79, 70, 229, 0.8)'; // Indigo (Symbolic)
    const rand = Math.random();
    if (rand < 0.3) {
      color = 'rgba(124, 58, 237, 0.8)'; // Violet (Neural)
    } else if (rand > 0.85) {
      color = 'rgba(0, 212, 255, 0.9)'; // Cyan (Grounded)
    }
    
    nodes.push({ x, y, z, color, size: Math.random() * 4 + 3 });
  }
  
  function draw() {
    ctx.clearRect(0, 0, size, size);
    const cx = size / 2;
    const cy = size / 2;
    
    // Dynamic angle adjustment based on mouse position
    const currentAngleY = angleY + mouseX * 0.001;
    const currentAngleX = mouseY * 0.001;
    
    // Rotate nodes
    const cosY = Math.cos(currentAngleY);
    const sinY = Math.sin(currentAngleY);
    const cosX = Math.cos(currentAngleX);
    const sinX = Math.sin(currentAngleX);
    
    nodes.forEach(n => {
      // Y-axis rotation
      let x1 = n.x * cosY - n.z * sinY;
      let z1 = n.z * cosY + n.x * sinY;
      
      // X-axis rotation
      let y2 = n.y * cosX - z1 * sinX;
      let z2 = z1 * cosX + n.y * sinX;
      
      n.x = x1;
      n.y = y2;
      n.z = z2;
      
      // Projection scale factor
      const depth = 280;
      const scale = depth / (depth + z2);
      n.projX = cx + x1 * scale;
      n.projY = cy + y2 * scale;
      n.projSize = n.size * scale;
    });
    
    // Draw edges between close nodes
    ctx.lineWidth = 0.5;
    for (let i = 0; i < nodeCount; i++) {
      for (let j = i + 1; j < nodeCount; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dz = nodes[i].z - nodes[j].z;
        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
        
        if (dist < 110) {
          // Fade edge based on depth of both nodes
          const avgZ = (nodes[i].z + nodes[j].z) / 2;
          const alpha = Math.max(0.02, 0.15 * (160 - avgZ) / 320);
          ctx.strokeStyle = `rgba(124, 58, 237, ${alpha})`;
          ctx.beginPath();
          ctx.moveTo(nodes[i].projX, nodes[i].projY);
          ctx.lineTo(nodes[j].projX, nodes[j].projY);
          ctx.stroke();
        }
      }
    }
    
    // Draw nodes
    nodes.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.projX, n.projY, n.projSize, 0, Math.PI * 2);
      ctx.fillStyle = n.color;
      ctx.shadowBlur = n.projSize * 2;
      ctx.shadowColor = n.color;
      ctx.fill();
      ctx.shadowBlur = 0; // reset
    });
    
    requestAnimationFrame(draw);
  }
  
  draw();
}

// 3. Console input and streaming trace logs
function initQueryConsole() {
  const input = document.getElementById('console-input');
  const body = document.getElementById('terminal-body');
  
  if (!input || !body) return;
  
  input.addEventListener('keypress', async (e) => {
    if (e.key === 'Enter') {
      const q = input.value.strip ? input.value.strip() : input.value.trim();
      if (!q) return;
      
      input.disabled = true;
      body.innerHTML = '';
      
      try {
        const response = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const data = await response.json();
        
        if (data.success) {
          await renderTraceLogs(data.trace);
        } else {
          body.innerHTML = `<div style="color: var(--accent-red); font-family: var(--font-mono);">&gt; Process error: ${data.detail || 'Failed query execution.'}</div>`;
          input.disabled = false;
        }
      } catch (err) {
        console.error(err);
        body.innerHTML = `<div style="color: var(--accent-red); font-family: var(--font-mono);">&gt; Network connection refused.</div>`;
        input.disabled = false;
      }
    }
  });
  
  // Helper to type out lines mimicking real-time cognitive trace
  async function typeText(element, text, speed = 20) {
    return new Promise(resolve => {
      let idx = 0;
      element.innerHTML = '';
      function step() {
        if (idx < text.length) {
          element.innerHTML += text.charAt(idx);
          idx++;
          setTimeout(step, speed);
        } else {
          resolve();
        }
      }
      step();
    });
  }
  
  async function renderTraceLogs(trace) {
    for (let step of trace) {
      // Create Step Box element
      const box = document.createElement('div');
      box.className = `trace-step step-${step.stage}`;
      
      const title = document.createElement('div');
      title.className = 'step-header';
      title.innerText = `[STAGE ${step.stage}] ${step.title}`;
      
      const content = document.createElement('div');
      content.className = 'step-text';
      
      box.appendChild(title);
      box.appendChild(content);
      body.appendChild(box);
      
      // Scroll terminal
      body.scrollTop = body.scrollHeight;
      
      // Activate step visibility
      await new Promise(r => setTimeout(r, 100));
      box.classList.add('active');
      
      // If it contains data representation (like confidence or results)
      if (step.data) {
        let msg = step.message;
        if (step.data.candidate) {
          msg += ` (Confidence: ${step.data.confidence}%)`;
        }
        await typeText(content, msg, 15);
        
        // Add visual metrics charts if present
        if (step.data.confidence) {
          const progress = document.createElement('div');
          progress.className = 'metric-meter';
          const fill = document.createElement('div');
          fill.className = 'meter-fill';
          progress.appendChild(fill);
          box.appendChild(progress);
          
          // Force reflow and fill meter
          setTimeout(() => {
            fill.style.width = `${step.data.confidence}%`;
          }, 50);
        }
      } else {
        await typeText(content, step.message, 15);
      }
      
      // Add slight pause between stages to mimic thought latency
      await new Promise(r => setTimeout(r, 800));
    }
    
    // Re-enable console input
    input.disabled = false;
    input.value = '';
    input.focus();
  }
}
