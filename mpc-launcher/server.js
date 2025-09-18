const express = require('express');
const cors = require('cors');
const { spawn, exec } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3001; // Different from React (3000)

// Middleware
app.use(cors({
  origin: ['http://localhost:3000', 'http://127.0.0.1:3000'],
  credentials: true
}));
app.use(express.json());

// Specific MPC path
const MPC_PATHS = [
  'C:\\Program Files (x86)\\K-Lite Codec Pack\\MPC-HC64\\mpc-hc64.exe'
];

// Find available MPC executable
function findMPCExecutable() {
  for (const mpcPath of MPC_PATHS) {
    try {
      if (fs.existsSync(mpcPath)) {
        return mpcPath;
      }
    } catch (error) {
      continue;
    }
  }
  return null;
}

// Launch MPC with URL and precise frame seeking
function launchMPC(youtubeUrl, timestamp, frameInfo, callback) {
  const mpcPath = findMPCExecutable();
  
  if (!mpcPath) {
    return callback(new Error('MPC-HC or MPC-BE not found. Please install MPC or add it to PATH.'));
  }

  // Create URL with timestamp for YouTube
  const urlWithTimestamp = timestamp > 0 ? `${youtubeUrl}&t=${timestamp}s` : youtubeUrl;

  console.log(`🎬 Launching MPC: ${mpcPath}`);
  console.log(`📺 URL: ${urlWithTimestamp}`);
  console.log(`⏰ Timestamp: ${timestamp}s (${Math.floor(timestamp/60)}:${(timestamp%60).toString().padStart(2,'0')})`);

  // Calculate precise timestamp from frame order (if available)
  let seekMs = timestamp * 1000;

  if (frameInfo) {
    console.log(`🎯 Frame Order: ${frameInfo.frameOrder} (${frameInfo.videoId})`);
    console.log(`📊 FPS: ${frameInfo.fps}`);

    const preciseTimestamp = frameInfo.frameOrder / frameInfo.fps;
    seekMs = Math.floor(preciseTimestamp * 1000);

    console.log(`✨ Precise Timestamp: ${preciseTimestamp.toFixed(3)}s (${seekMs}ms)`);
  }

  // Launch MPC with optimal method (method 1 - most accurate)
  const mpcProcess = spawn(mpcPath, [urlWithTimestamp, '/start', seekMs.toString()], {
    detached: true,
    stdio: 'ignore'
  });

  mpcProcess.unref();

  mpcProcess.on('error', (error) => {
    console.error('❌ MPC launch failed:', error.message);
    callback(new Error(`Failed to launch MPC: ${error.message}`));
  });

  mpcProcess.on('spawn', () => {
    console.log('✅ MPC launched successfully');
    callback(null, {
      success: true,
      method: 'method_1',
      path: mpcPath,
      timestamp: timestamp,
      seekMs: seekMs
    });
  });
}

// API Routes
app.post('/launch-mpc', (req, res) => {
  const { youtubeUrl, timestamp = 0, videoTitle, frameInfo } = req.body;

  if (!youtubeUrl) {
    return res.status(400).json({
      error: 'YouTube URL is required'
    });
  }

  console.log(`\n🚀 MPC Launch: ${videoTitle || 'Unknown'}`);
  console.log(`📺 URL: ${youtubeUrl}`);

  if (frameInfo) {
    console.log(`🎯 ${frameInfo.videoId} → Frame ${frameInfo.frameOrder} @ ${frameInfo.fps}fps`);
  } else {
    console.log(`⏰ Timestamp: ${timestamp}s`);
  }

  launchMPC(youtubeUrl, timestamp, frameInfo, (error, result) => {
    if (error) {
      console.error('❌ MPC Launch Failed:', error.message);
      return res.status(500).json({ 
        error: error.message,
        suggestions: [
          'Install MPC-HC from: https://mpc-hc.org/',
          'Install MPC-BE from: https://sourceforge.net/projects/mpcbe/',
          'Add MPC to your system PATH',
          'Ensure yt-dlp is configured in MPC'
        ]
      });
    }

    console.log(`✅ MPC Launched Successfully:`, result);
    res.json({
      success: true,
      message: 'MPC launched successfully',
      mpcPath: result.path,
      method: result.method,
      url: youtubeUrl,
      timestamp: timestamp
    });
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  const mpcPath = findMPCExecutable();
  res.json({
    status: 'running',
    mpcAvailable: !!mpcPath,
    mpcPath: mpcPath,
    port: PORT
  });
});

// Start server
app.listen(PORT, '127.0.0.1', () => {
  const mpcPath = findMPCExecutable();
  console.log(`\n🎬 MPC Launcher Service - Ready on http://127.0.0.1:${PORT}`);
  console.log(`${mpcPath ? '✅' : '❌'} MPC: ${mpcPath || 'Not Found'}`);
  if (mpcPath) {
    console.log(`🚀 Ready for precise frame launching!`);
  }
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n👋 MPC Launcher Service shutting down...');
  process.exit(0);
});