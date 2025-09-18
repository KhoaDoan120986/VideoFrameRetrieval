/**
 * Video Event Retrieval AI Challenge 2025
 * Multimodal search interface with text and image search capabilities
 * Supports TRAKE, Q&A, and Textual KIS modes
 */

import React, { useState, useEffect } from "react";
import { FaArrowUp, FaSearch } from "react-icons/fa";
import { FiSettings } from "react-icons/fi";
import "./App.css";

const resizeObserverErrHandler = e => {
  if (e.message === "ResizeObserver loop completed with undelivered notifications.") {
    return;
  }
  console.error(e);
};
window.addEventListener("error", resizeObserverErrHandler);

function App() {
  // Search State
  const [query, setQuery] = useState("");
  const [searchType, setSearchType] = useState("text");
  const [searchMode, setSearchMode] = useState("hybrid");
  const [captionMode, setCaptionMode] = useState("bge");
  const [alpha, setAlpha] = useState(0.5);
  const [isSearching, setIsSearching] = useState(false);
  const [searchTime, setSearchTime] = useState(null);
  const [topK, setTopK] = useState(10);

  // Results State
  const [results, setResults] = useState([]);
  const [pageIndex, setPageIndex] = useState(0);
  const [selectedItems, setSelectedItems] = useState(new Set());

  // Backend State
  const [backendUrl, setBackendUrl] = useState("");
  const [isConnected, setIsConnected] = useState(false);

  // Image Search State
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  // App Modes State
  const [appMode, setAppMode] = useState("textual-kis");
  const [searchModeType, setSearchModeType] = useState("normal"); // "normal" or "temporal"
  const [frameAnswers, setFrameAnswers] = useState({});
  const [csvFileName, setCsvFileName] = useState("selected_results");

  // Video Player State
  const [loadingVideo, setLoadingVideo] = useState(null);
  const [videoPlayer, setVideoPlayer] = useState(null);
  const [timeInput, setTimeInput] = useState("");
  const [calculatedFrameOrder, setCalculatedFrameOrder] = useState(null);
  const [videoPlayerPreference, setVideoPlayerPreference] = useState("embedded"); // "embedded" or "mpc"
  
  // Video Filter State
  const [videoFilter, setVideoFilter] = useState("all"); // "all" or specific video ID
  
  // Sidebar Video Keyframes Viewer State
  const [selectedVideoForViewing, setSelectedVideoForViewing] = useState(""); // Video ID to view keyframes
  const [videoKeyframes, setVideoKeyframes] = useState([]); // Keyframes from selected video
  const [loadingVideoKeyframes, setLoadingVideoKeyframes] = useState(false);
  
  // Video Browse State (right side display)
  const [videoBrowseMode, setVideoBrowseMode] = useState(false);
  const [browsedVideoId, setBrowsedVideoId] = useState("");
  const [browseCurrentPage, setBrowseCurrentPage] = useState(0);
  const [browseZoomedFrame, setBrowseZoomedFrame] = useState(null);
  
  // Zoom State for TRAKE frames
  const [zoomedFrame, setZoomedFrame] = useState(null);
  // Temporal Search State (available for all modes)
  const [temporalSearch, setTemporalSearch] = useState({
    events: [""],
    results: [],
    currentPage: 0,
    isActive: false,
    searchMode: "progressive"  // "progressive" or "consolidated"
  });

  // TRAKE Mode State (original design)
  const [trakeMode, setTrakeMode] = useState({
    selectedVideo: null,
    videoFrames: [],
    eventSequences: [],
    currentSequence: [],
    currentPage: 0,
    framesPerPage: 10
  });
  // Constants
  const pageSize = 8;

  // Helper function to determine batch configuration
  const getBatchConfig = (videoId) => {
    const videoPrefix = videoId.charAt(0);
    
    if (videoPrefix === 'L') {
      // Batch 1: L-prefixed videos (L21_V001, L22_V002, etc.)
      return {
        keyframesDir: 'keyframes',
        mediaInfoDir: 'media-info-aic25-b1/media-info',
        isKPrefixed: false
      };
    } else if (videoPrefix === 'K') {
      // Batch 2: K-prefixed videos (K01_V001, K02_V002, etc.)
      return {
        keyframesDir: 'keyframes-b2',
        mediaInfoDir: 'media-info-aic25-b2/media-info', 
        isKPrefixed: true
      };
    } else {
      // Default to batch 1 for unknown formats
      return {
        keyframesDir: 'keyframes',
        mediaInfoDir: 'media-info-aic25-b1/media-info',
        isKPrefixed: false
      };
    }
  };

  async function fetchSearchResults(query, imageFile = null) {
    if (!backendUrl.trim()) {
      alert("Please enter the Kaggle backend URL first!");
      return [];
    }

    try {
      let response;

      if (searchType === "image" && imageFile) {
        // Image search using FormData
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('topK', topK.toString());
        formData.append('mode', 'image');
        formData.append('query', '');

        response = await fetch(`${backendUrl}/search`, {
          method: "POST",
          headers: {
            "ngrok-skip-browser-warning": "true"
          },
          body: formData,
        });
      } else {
        // Text search using FormData (backend expects Form parameters)
        const formData = new FormData();
        formData.append('query', query);
        formData.append('topK', topK.toString());
        formData.append('mode', searchMode);
        formData.append('caption_mode', captionMode);
        formData.append('alpha', alpha.toString());

        response = await fetch(`${backendUrl}/search`, {
          method: "POST",
          headers: {
            "ngrok-skip-browser-warning": "true"
          },
          body: formData,
        });
      }

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
      const data = await response.json();
      setIsConnected(true);
      return data.results || [];
    } catch (error) {
      setIsConnected(false);
      alert(`Search failed: ${error.message}\nPlease check your backend URL.`);
      return [];
    }
  }


  const toggleItemSelection = (videoId) => {
    const newSelection = new Set(selectedItems);
    if (newSelection.has(videoId)) {
      newSelection.delete(videoId);
    } else {
      newSelection.add(videoId);
    }
    setSelectedItems(newSelection);
  };

  const selectAllCurrentPage = () => {
    const newSelection = new Set(selectedItems);
    currentPageData.forEach(item => {
      newSelection.add(item.videoId);
    });
    setSelectedItems(newSelection);
  };

  const clearAllSelection = () => {
    setSelectedItems(new Set());
  };



  // Simple manual time to frame conversion with milliseconds support
  const handleTimeToFrameConversion = () => {
    if (!timeInput.trim()) {
      setCalculatedFrameOrder(null);
      return;
    }

    if (!videoPlayer) {
      alert('No video is loaded');
      return;
    }

    try {
      // Parse time input (supports MM:SS.mmm, MM:SS, or just seconds)
      let totalSeconds = 0;
      let displayTime = timeInput;
      
      if (timeInput.includes(':')) {
        // Handle MM:SS.mmm or MM:SS format
        const timeParts = timeInput.split(':');
        if (timeParts.length === 2) {
          const minutes = parseInt(timeParts[0]) || 0;
          
          // Handle seconds with possible milliseconds
          const secondsPart = timeParts[1];
          let seconds = 0;
          
          if (secondsPart.includes('.')) {
            // Has milliseconds: SS.mmm
            seconds = parseFloat(secondsPart) || 0;
          } else {
            // Just seconds: SS
            seconds = parseInt(secondsPart) || 0;
          }
          
          totalSeconds = minutes * 60 + seconds;
        }
      } else {
        // Just seconds (with possible decimals)
        totalSeconds = parseFloat(timeInput) || 0;
      }
      
      // Calculate frame order using video FPS with precise milliseconds
      const fps = videoPlayer.fps || 25;
      const frameOrder = Math.floor(totalSeconds * fps);
      
      // For display, format the time properly
      if (!timeInput.includes(':') && totalSeconds >= 60) {
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = (totalSeconds % 60).toFixed(3);
        displayTime = `${minutes}:${seconds.padStart(6, '0')}`;
      }
      
      setCalculatedFrameOrder({
        inputTime: timeInput,
        displayTime,
        totalSeconds: totalSeconds.toFixed(3),
        fps,
        frameOrder,
        preciseCalculation: `${totalSeconds.toFixed(3)}s × ${fps} FPS = ${(totalSeconds * fps).toFixed(3)} → ${frameOrder}`
      });
      
    } catch (error) {
      console.error('Error converting time to frame:', error);
      alert('Invalid time format. Use MM:SS.mmm, MM:SS, or seconds with decimals.');
    }
  };


  const handleKeyframeClick = async (videoId) => {
    setLoadingVideo(videoId);
    try {
      const parts = videoId.split('_');
      if (parts.length < 3) {
        alert('Invalid video ID format');
        return;
      }

      const batch = parts[0];
      const videoNum = parts[1];
      const keyframeOrder = parseInt(parts[2]);
      const videoFile = `${batch}_${videoNum}`;

      // Get batch configuration to determine correct media info path
      const batchConfig = getBatchConfig(videoId);
      const mediaInfoPath = `/${batchConfig.mediaInfoDir}/${videoFile}.json`;
      const response = await fetch(mediaInfoPath);
      if (!response.ok) {
        throw new Error(`Could not load video info for ${videoFile}`);
      }

      const videoInfo = await response.json();
      const youtubeUrl = videoInfo.watch_url;
      const videoDuration = videoInfo.length;
      const videoFPS = videoInfo.fps || 25;

      let finalTimestamp = Math.floor(keyframeOrder / videoFPS);
      finalTimestamp = Math.max(0, Math.min(finalTimestamp, videoDuration - 1));

      const videoIdMatch = youtubeUrl.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/);
      const youtubeVideoId = videoIdMatch ? videoIdMatch[1] : null;

      if (youtubeVideoId) {
        if (videoPlayerPreference === "mpc") {
          // Open in MPC with precise frame info
          openInMPC(youtubeUrl, finalTimestamp, videoInfo.title || `Video ${videoFile}`, {
            frameOrder: keyframeOrder,
            fps: videoFPS,
            videoId: videoId
          });
        } else {
          // Use embedded player
          setVideoPlayer({
            videoId: youtubeVideoId,
            timestamp: finalTimestamp,
            title: videoInfo.title || `Video ${videoFile}`,
            keyframe: videoId,
            fps: videoFPS,
            videoDuration: videoDuration
          });
        }
        
        // Reset calculator when opening new video  
        setTimeInput("");
        setCalculatedFrameOrder(null);
      } else {
        throw new Error('Could not extract YouTube video ID');
      }

    } catch (error) {
      console.error('Error opening video:', error);
      alert(`Error opening video: ${error.message}`);
    } finally {
      setLoadingVideo(null);
    }
  };

  // Direct MPC Integration Function via Local Service
  const openInMPC = async (youtubeUrl, timestamp, videoTitle = '', frameInfo = null) => {
    try {
      console.log(`🎬 Opening MPC: ${videoTitle}`);
      console.log(`📺 URL: ${youtubeUrl}`);
      console.log(`⏰ Timestamp: ${timestamp}s`);
      if (frameInfo) {
        console.log(`🎯 Frame Order: ${frameInfo.frameOrder} (${frameInfo.videoId})`);
        console.log(`🎞️ FPS: ${frameInfo.fps}`);
      }

      const response = await fetch('http://127.0.0.1:3001/launch-mpc', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          youtubeUrl: youtubeUrl,
          timestamp: timestamp,
          videoTitle: videoTitle,
          frameInfo: frameInfo
        }),
      });

      const result = await response.json();

      if (response.ok && result.success) {
        console.log('✅ MPC launched successfully:', result);
        // Optional: Show brief success notification
        setTimeout(() => {
          alert(`✅ MPC launched successfully!\n\n🎬 Player: ${result.mpcPath.split('\\').pop()}\n⏰ Timestamp: ${timestamp}s`);
        }, 500);
      } else {
        throw new Error(result.error || 'Failed to launch MPC');
      }

    } catch (error) {
      console.error('❌ MPC launch failed:', error);
      
      // Check if it's a connection error (service not running)
      if (error.message.includes('fetch')) {
        const startService = window.confirm(
          `❌ MPC Launcher Service not running!\n\n` +
          `To use direct MPC integration:\n` +
          `1. Open terminal in: mpc-launcher folder\n` +
          `2. Run: npm install\n` +
          `3. Run: npm start\n\n` +
          `Click OK for manual MPC instructions, or Cancel to use embedded player.`
        );
        
        if (startService) {
          // Fallback to manual instructions
          const urlWithTimestamp = `${youtubeUrl}&t=${timestamp}s`;
          const confirmation = window.confirm(
            `📋 Manual MPC Instructions:\n\n` +
            `1. Open MPC-HC or MPC-BE\n` +
            `2. Press Ctrl+U (Open URL)\n` +
            `3. Paste this URL:\n${urlWithTimestamp}\n\n` +
            `Click OK to copy URL to clipboard.`
          );
          
          if (confirmation) {
            navigator.clipboard.writeText(urlWithTimestamp).then(() => {
              alert('✅ YouTube URL copied to clipboard!');
            }).catch(() => {
              window.prompt('Copy this URL:', urlWithTimestamp);
            });
          }
        }
      } else {
        // MPC not found or other error
        alert(`❌ Could not launch MPC: ${error.message}\n\nPlease ensure MPC-HC or MPC-BE is installed.`);
      }
    }
  };

  const updateFrameAnswer = (videoId, answer) => {
    setFrameAnswers(prev => ({
      ...prev,
      [videoId]: answer
    }));
  };

  const fetchVideoFrames = async (videoId) => {
    const parts = videoId.split('_');
    const batch = parts[0];
    const videoNum = parts[1];
    const baseVideoId = `${batch}_${videoNum}`;

    try {
      // Load the keyframes index based on video batch
      const batchConfig = getBatchConfig(videoId);
      const indexResponse = await fetch(`/${batchConfig.mediaInfoDir}/keyframes_index.json`);

      if (indexResponse.ok) {
        const keyframesIndex = await indexResponse.json();
        const imagePaths = keyframesIndex[baseVideoId];

        if (imagePaths && imagePaths.length > 0) {
          // Convert image paths to frame objects
          const frames = imagePaths.map(imagePath => {
            // Extract frame number from path (e.g., /keyframes/L21/L21_V008/L21_V008_1200.jpg -> 1200)
            const filename = imagePath.split('/').pop();
            const frameMatch = filename.match(/_(\d+)\.jpg$/);
            const frameNum = frameMatch ? parseInt(frameMatch[1]) : 0;

            return {
              videoId: `${baseVideoId}_${frameNum}`,
              frameNum: frameNum,
              imagePath: imagePath
            };
          });

          // Sort by frame number
          frames.sort((a, b) => a.frameNum - b.frameNum);
          return frames;
        }
      }
    } catch (error) {
      console.error('Error loading keyframes index:', error);
    }

    return [];
  };

  const selectVideoForTrake = async (videoId) => {
    const parts = videoId.split('_');
    const baseVideoId = `${parts[0]}_${parts[1]}`;

    setTrakeMode(prev => ({
      ...prev,
      selectedVideo: baseVideoId,
      videoFrames: [],
      currentSequence: [],
      currentPage: 0
    }));

    const frames = await fetchVideoFrames(videoId);

    setTrakeMode(prev => ({
      ...prev,
      videoFrames: frames
    }));
  };

  const goToTrakePage = (pageIndex) => {
    const totalPages = Math.ceil(trakeMode.videoFrames.length / 10);
    let newPage = pageIndex;

    if (newPage < 0) newPage = 0;
    if (newPage >= totalPages) newPage = totalPages - 1;

    setTrakeMode(prev => ({
      ...prev,
      currentPage: newPage
    }));
  };

  const addFrameToSequence = (frameNum) => {
    setTrakeMode(prev => ({
      ...prev,
      currentSequence: [...prev.currentSequence, frameNum].sort((a, b) => a - b)
    }));
  };

  const removeFrameFromSequence = (frameNum) => {
    setTrakeMode(prev => ({
      ...prev,
      currentSequence: prev.currentSequence.filter(f => f !== frameNum)
    }));
  };

  const saveEventSequence = () => {
    if (trakeMode.currentSequence.length === 0) {
      alert("Please select at least one frame for the event sequence.");
      return;
    }

    setTrakeMode(prev => ({
      ...prev,
      eventSequences: [
        ...prev.eventSequences,
        {
          videoId: prev.selectedVideo,
          frames: [...prev.currentSequence]
        }
      ],
      currentSequence: []
    }));
  };

  // Temporal Search Functions (available for all modes)
  const handleTemporalEventChange = (index, value) => {
    setTemporalSearch(prev => ({
      ...prev,
      events: prev.events.map((event, i) => 
        i === index ? value : event
      )
    }));
  };

  const addTemporalEvent = () => {
    setTemporalSearch(prev => ({
      ...prev,
      events: [...prev.events, ""]
    }));
  };

  const removeTemporalEvent = (index) => {
    if (temporalSearch.events.length <= 1) return;
    setTemporalSearch(prev => ({
      ...prev,
      events: prev.events.filter((_, i) => i !== index)
    }));
  };


  const getTemporalSearchParams = () => {
    // Adapt search parameters based on app mode
    switch (appMode) {
      case "qa":
        return {
          topK: Math.min(topK * 0.8, 50), // Fewer results for detailed Q&A analysis
          searchMode: "hybrid", // Balanced approach for questions
          alpha: 0.7 // More text-focused for questions
        };
      case "trake":
        return {
          topK: topK, // Use standard topK value, no enhancement
          searchMode: searchMode, // Use current mode setting
          alpha: alpha // Use current alpha setting
        };
      default: // textual-kis
        return {
          topK: topK,
          searchMode: searchMode,
          alpha: alpha
        };
    }
  };

  const handleTemporalSearch = async () => {
    const validEvents = temporalSearch.events.filter(event => event.trim() !== "");
    if (validEvents.length === 0) {
      alert("Please enter at least one temporal event to search.");
      return;
    }

    // No mode-specific validation - allow any number of events for all modes

    setIsSearching(true);
    const startTime = Date.now();
    
    // Reset video filter for new temporal search
    setVideoFilter("all");
    
    try {
      // Get mode-specific parameters
      const params = getTemporalSearchParams();
      
      const formData = new FormData();
      formData.append('events', JSON.stringify(validEvents));
      formData.append('topK', params.topK.toString());
      formData.append('mode', params.searchMode);
      formData.append('caption_mode', captionMode);
      formData.append('alpha', params.alpha.toString());
      formData.append('search_mode', temporalSearch.searchMode);

      const response = await fetch(`${backendUrl}/temporal_search`, {
        method: "POST",
        headers: {
          "ngrok-skip-browser-warning": "true"
        },
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      const endTime = Date.now();
      const searchDuration = ((endTime - startTime) / 1000).toFixed(2);
      
      // Check if the response has the expected structure
      if (!data || (!data.results && !data.error)) {
        throw new Error("Invalid response structure from backend");
      }
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      let mappedResults;
      
      if (temporalSearch.searchMode === "consolidated") {
        // Consolidated mode: Handle video timeline results from backend
        mappedResults = (data.results || []).map(videoResult => {
          // Get the best frame sequence (first one from beam search)
          const bestSequence = videoResult.frame_sequence[0] || [];
          const firstFrame = bestSequence[0];
          
          if (!firstFrame) return null;
          
          const firstFrameId = firstFrame.id;
          const parts = firstFrameId.split('_');
          const batch = parts[0];
          const videoNumber = parts[1];
          const baseVideoId = `${batch}_${videoNumber}`;
          const batchConfig = getBatchConfig(firstFrameId);
          const imagePath = `/${batchConfig.keyframesDir}/${batch}/${baseVideoId}/${firstFrameId}.jpg`;

          // Create timeline data structure with individual frame scores
          const timelineData = {
            video_id: videoResult.video,
            score: videoResult.score,
            image: bestSequence.map(frame => frame.id), // Array of frame IDs
            frameScores: bestSequence.map(frame => frame.score), // Individual frame scores
            frames: bestSequence // Full frame objects with id and score
          };

          return {
            videoId: firstFrameId, // Primary frame ID for display
            image: imagePath,
            caption: `${videoResult.video} | Timeline Score: ${videoResult.score.toFixed(3)}`,
            videoTimeline: timelineData, // Store timeline data with individual scores
            isConsolidated: true
          };
        }).filter(result => result !== null);
      } else {
        // Progressive mode: Display identical to consolidated mode with complete timeline
        // Backend returns array of video results with frame sequences
        
        let processedResults;
        if (appMode === "trake") {
          processedResults = (data.results || []).slice(0, params.topK);
        } else {
          processedResults = (data.results || []).slice(0, params.topK);
        }
        
        mappedResults = processedResults.map(videoResult => {
          // Get the best frame sequence (first one from beam search)
          const bestSequence = videoResult.frame_sequence[0] || [];
          const firstFrame = bestSequence[0]; // Use first frame for primary display
          
          if (!firstFrame) return null;
          
          // Create timeline data structure with individual frame scores (same as consolidated)
          const timelineData = {
            video_id: videoResult.video,
            score: videoResult.score,
            image: bestSequence.map(frame => frame.id), // Array of frame IDs
            frameScores: bestSequence.map(frame => frame.score), // Individual frame scores
            frames: bestSequence
          };

          const firstFrameId = firstFrame.id;
          const parts = firstFrameId.split('_');
          const batch = parts[0];
          const videoNumber = parts[1];
          const baseVideoId = `${batch}_${videoNumber}`;
          const batchConfig = getBatchConfig(firstFrameId);
          const imagePath = `/${batchConfig.keyframesDir}/${batch}/${baseVideoId}/${firstFrameId}.jpg`;

          return {
            videoId: firstFrameId, // Primary frame ID for display
            image: imagePath,
            caption: `${videoResult.video} | Timeline Score: ${videoResult.score.toFixed(3)}`,
            videoTimeline: timelineData, // Store complete timeline data with individual scores
            isConsolidated: true // Progressive mode with same timeline display as consolidated
          };
        }).filter(result => result !== null);
      }

      // Store all intermediate results for potential future use
      const allResults = data.results || [];

      setTemporalSearch(prev => ({
        ...prev,
        results: mappedResults,
        allResults: allResults, // Store all intermediate results
        currentPage: 0,
        isActive: true,
        searchInfo: data.search_info || {},
        events: validEvents // Store the actual events used
      }));
      setResults(mappedResults);
      setPageIndex(0);
      setSearchTime(searchDuration);

      // Mode-specific post-processing
      if (appMode === "qa") {
        // Pre-select all results for Q&A mode for easier answer input
        const newSelection = new Set(mappedResults.map(item => item.videoId));
        setSelectedItems(newSelection);
      }
      
    } catch (error) {
      console.error('Temporal search error:', error);
      alert(`Temporal search failed: ${error.message}\nPlease check your backend URL.`);
    } finally {
      setIsSearching(false);
    }
  };

  const exitTemporalSearch = () => {
    setTemporalSearch(prev => ({
      ...prev,
      isActive: false,
      results: [],
      searchMode: "progressive"
    }));
    setResults([]);
    setPageIndex(0);
  };

  const deleteEventSequence = (index) => {
    setTrakeMode(prev => ({
      ...prev,
      eventSequences: prev.eventSequences.filter((_, i) => i !== index)
    }));
  };

  // Helper function to extract frame timestamp from full frame ID
  const extractFrameTimestamp = (frameId) => {
    const parts = frameId.split('_');
    return parts.length >= 3 ? parts[2] : frameId;
  };

  // Helper function to create CSV blob and download
  const downloadCSVFile = (csvData, filename) => {
    const csvContent = csvData.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${csvFileName || filename}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  };

  // Generate CSV data for Timeline View (both consolidated and progressive) results
  const generateTimelineCSV = (results) => {
    const csvData = [];
    results.forEach(result => {
      if (result.videoTimeline && selectedItems.has(result.videoId)) {
        const videoId = result.videoTimeline.video_id;
        const frameSequence = result.videoTimeline.image.map(extractFrameTimestamp);
        csvData.push(`${videoId}, ${frameSequence.join(', ')}`);
      }
    });
    return csvData;
  };

  // Generate CSV data for Sequential Search results
  const generateSequentialCSV = (results) => {
    const csvData = [];
    results.forEach(result => {
      const parts = result.videoId.split('_');
      if (parts.length >= 3) {
        const videoFile = `${parts[0]}_${parts[1]}`;
        const frameNum = parts[2];
        
        if (appMode === "qa") {
          const answer = frameAnswers[result.videoId] || "";
          csvData.push(`${videoFile}, ${frameNum}, "${answer.replace(/"/g, '""')}"`);
        } else {
          csvData.push(`${videoFile}, ${frameNum}`);
        }
      }
    });
    return csvData;
  };

  const downloadCSV = () => {
    // Handle temporal search results
    if (searchModeType === "temporal" && temporalSearch.isActive && temporalSearch.results.length > 0) {
      const hasTimelineResults = temporalSearch.results.some(result => 
        result.videoTimeline
      );
      
      if (hasTimelineResults) {
        const csvData = generateTimelineCSV(temporalSearch.results);
        downloadCSVFile(csvData, 'timeline_sequences');
      } else {
        const csvData = generateSequentialCSV(temporalSearch.results);
        downloadCSVFile(csvData, 'temporal_search_results');
      }
      return;
    }

    if (appMode === "trake") {
      if (trakeMode.eventSequences.length === 0) {
        alert("Please create at least one event sequence to download.");
        return;
      }

      const csvData = [];
      trakeMode.eventSequences.forEach(sequence => {
        const framesList = sequence.frames.join(', ');
        csvData.push(`${sequence.videoId}, ${framesList}`);
      });

      const csvContent = csvData.join('\n');
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${csvFileName || 'trake_event_sequences'}.csv`;
      link.click();
      window.URL.revokeObjectURL(url);
      return;
    }

    if (selectedItems.size === 0) {
      alert("Please select at least one item to download.");
      return;
    }

    const csvData = [];

    selectedItems.forEach(videoId => {
      const parts = videoId.split('_');
      if (parts.length >= 3) {
        const batch = parts[0];
        const videoNum = parts[1];
        const frameNum = parts[2];
        const videoFile = `${batch}_${videoNum}`;

        if (appMode === "qa") {
          const answer = frameAnswers[videoId] || "";
          csvData.push(`${videoFile}, ${frameNum}, "${answer.replace(/"/g, '""')}"`);
        } else {
          csvData.push(`${videoFile}, ${frameNum}`);
        }
      }
    });

    const csvContent = csvData.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const defaultName = appMode === "qa" ? "qa_selected_results" : "textual_kis_selected_results";
    link.download = `${csvFileName || defaultName}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  };

  const handleImageUpload = (file) => {
    if (file && file.type.startsWith('image/')) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    } else {
      alert('Please select a valid image file.');
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    handleImageUpload(file);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const clearImage = () => {
    setImageFile(null);
    setImagePreview(null);
  };

  // Video Filter Functions
  const getUniqueVideos = (results) => {
    const videos = new Set();
    results.forEach(item => {
      if (item.videoId) {
        const parts = item.videoId.split('_');
        if (parts.length >= 2) {
          const videoId = `${parts[0]}_${parts[1]}`; // e.g., "L01_V001"
          videos.add(videoId);
        }
      }
    });
    return Array.from(videos).sort();
  };

  const filterResultsByVideo = (results, filter) => {
    if (filter === "all") return results;
    
    return results.filter(item => {
      if (!item.videoId) return false;
      const parts = item.videoId.split('_');
      if (parts.length >= 2) {
        const videoId = `${parts[0]}_${parts[1]}`;
        return videoId === filter;
      }
      return false;
    });
  };

  const handleVideoFilterChange = (newFilter) => {
    setVideoFilter(newFilter);
    setPageIndex(0); // Reset to first page when filter changes
    if (searchModeType === "temporal" && temporalSearch.isActive) {
      setTemporalSearch(prev => ({
        ...prev,
        currentPage: 0
      }));
    }
  };

  // Function to load video keyframes using keyframes_index.json
  const loadVideoKeyframes = async (videoId) => {
    if (!videoId) {
      setVideoKeyframes([]);
      return;
    }

    setLoadingVideoKeyframes(true);
    try {
      // Load the keyframes index file based on video batch
      const batchConfig = getBatchConfig(videoId);
      const response = await fetch(`/${batchConfig.mediaInfoDir}/keyframes_index.json`);
      
      if (!response.ok) {
        throw new Error('Keyframes index not found');
      }
      
      const keyframesIndex = await response.json();
      
      // Get keyframes for the selected video
      const imagePaths = keyframesIndex[videoId];
      
      if (imagePaths && imagePaths.length > 0) {
        // Map image paths to frame objects
        const mappedKeyframes = imagePaths.map(imagePath => {
          // Extract frame number from path (e.g., /keyframes/L21/L21_V001/L21_V001_1200.jpg -> 1200)
          const filename = imagePath.split('/').pop();
          const frameMatch = filename.match(/_(\d+)\.jpg$/);
          const frameNum = frameMatch ? parseInt(frameMatch[1]) : 0;
          
          return {
            videoId: filename.replace('.jpg', ''), // L21_V001_1200
            image: imagePath,
            caption: `Frame at ${frameNum}ms from ${videoId}`,
            score: 1.0,
            timestamp: frameNum
          };
        });

        // Sort by timestamp for chronological order
        mappedKeyframes.sort((a, b) => a.timestamp - b.timestamp);
        
        setVideoKeyframes(mappedKeyframes);
      } else {
        // No keyframes found for this video
        setVideoKeyframes([]);
        console.log(`No keyframes found for video: ${videoId}`);
      }
    } catch (error) {
      console.error('Error loading video keyframes:', error);
      setVideoKeyframes([]);
    } finally {
      setLoadingVideoKeyframes(false);
    }
  };

  const handleVideoSelectionForViewing = (videoId) => {
    setSelectedVideoForViewing(videoId);
    if (videoId) {
      loadVideoKeyframes(videoId);
      // Also activate right-side video browse
      setBrowsedVideoId(videoId);
      setVideoBrowseMode(true);
      setBrowseCurrentPage(0);
    } else {
      setVideoKeyframes([]);
      setVideoBrowseMode(false);
      setBrowsedVideoId("");
    }
  };

  // Video Browse Functions
  const exitVideoBrowse = () => {
    setVideoBrowseMode(false);
    setBrowsedVideoId("");
    setBrowseCurrentPage(0);
    setBrowseZoomedFrame(null);
  };

  const goToBrowsePage = (newPageIndex) => {
    const browseFramesPerPage = 8;
    const totalBrowsePages = Math.ceil(videoKeyframes.length / browseFramesPerPage);
    
    if (newPageIndex < 0) newPageIndex = 0;
    else if (newPageIndex >= totalBrowsePages) newPageIndex = totalBrowsePages - 1;
    
    setBrowseCurrentPage(newPageIndex);
  };

  async function search() {
    // Route to appropriate search based on search mode type
    if (searchModeType === "temporal") {
      return handleTemporalSearch();
    }

    // Normal search validation
    if (searchType === "text" && !query.trim()) {
      setResults([]);
      setSearchTime(null);
      setSelectedItems(new Set());
      setFrameAnswers({});
      return;
    }

    if (searchType === "image" && !imageFile) {
      alert("Please upload an image for image search.");
      return;
    }

    setIsSearching(true);
    setSearchTime(null);
    setSelectedItems(new Set());
    setFrameAnswers({});
    setTrakeMode({
      selectedVideo: null,
      videoFrames: [],
      eventSequences: [],
      currentSequence: []
    });

    // Clear temporal search state when doing normal search
    setTemporalSearch({
      events: [""],
      results: [],
      currentPage: 0,
      isActive: false,
      searchMode: "progressive"
    });
    
    // Reset video filter for new search
    setVideoFilter("all");

    try {
      const startTime = performance.now();
      const data = await fetchSearchResults(query, imageFile);
      const endTime = performance.now();
      const timeTaken = ((endTime - startTime) / 1000).toFixed(2);

      const mappedResults = (data || []).map(item => {
        const videoId = item.image.trim();
        const parts = videoId.split('_');
        const batch = parts[0];
        const videoNumber = parts[1];
        const baseVideoId = `${batch}_${videoNumber}`;
        const batchConfig = getBatchConfig(videoId);
        const imagePath = `/${batchConfig.keyframesDir}/${batch}/${baseVideoId}/${videoId}.jpg`;

        return {
          ...item,
          videoId: videoId,
          image: imagePath
        };
      });

      setResults(mappedResults);
      setSearchTime(timeTaken);
      setPageIndex(0);
    } finally {
      setIsSearching(false);
    }
  }

  useEffect(() => {
    document.title = "AIC";
  }, []);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        if (browseZoomedFrame) {
          setBrowseZoomedFrame(null);
        } else if (zoomedFrame) {
          setZoomedFrame(null);
        } else if (videoPlayer) {
          setVideoPlayer(null);
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [videoPlayer, zoomedFrame, browseZoomedFrame]);

  // Pagination logic - separate for temporal search and regular search
  const temporalFrameSize = 8;
  const rawDataSource = (searchModeType === "temporal" && temporalSearch.isActive) ? temporalSearch.results : results;
  
  // Apply video filtering
  const currentDataSource = filterResultsByVideo(rawDataSource, videoFilter);
  const currentPageSize = (searchModeType === "temporal" && temporalSearch.isActive) ? temporalFrameSize : pageSize;
  
  const totalPages = Math.max(Math.ceil(currentDataSource.length / currentPageSize), 1);

  const currentPageData = currentDataSource.slice(
    ((searchModeType === "temporal" && temporalSearch.isActive) ? temporalSearch.currentPage : pageIndex) * currentPageSize,
    (((searchModeType === "temporal" && temporalSearch.isActive) ? temporalSearch.currentPage : pageIndex) + 1) * currentPageSize
  );

  const goToPage = (newPageIndex) => {
    if (newPageIndex < 0) newPageIndex = 0;
    else if (newPageIndex >= totalPages) newPageIndex = totalPages - 1;
    
    if (searchModeType === "temporal" && temporalSearch.isActive) {
      setTemporalSearch(prev => ({
        ...prev,
        currentPage: newPageIndex
      }));
    } else {
      setPageIndex(newPageIndex);
    }
  };

  const missingCount = currentPageSize - currentPageData.length;

  return (
    <div className="container">
      <div className="sidebar">
        <div className="settings-title"> <FiSettings /> SETTINGS </div>
        <div className="sidebar-content">
          <div className="url-section">
            <label className="url-label">
              Backend URL:
              <input
                className="url-input"
                type="text"
                placeholder="https://xxxxxxx.ngrok-free.app"
                value={backendUrl}
                onChange={(e) => setBackendUrl(e.target.value.trim())}
              />
            </label>
            <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
              <span className="status-dot"></span>
              {isConnected ? 'Connected' : 'Not Connected'}
            </div>
          </div>

          <label className="topk-label" >
            Top K: <input className="topk-input" type="number" min="10" max="100" step="1"
              value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
          </label>

          <div className="mode-selection-section">
            <label className="mode-label">App Mode:</label>
            <select
              className="mode-select"
              value={appMode}
              onChange={(e) => {
                setAppMode(e.target.value);
                setSelectedItems(new Set());
                setFrameAnswers({});
                setSearchModeType("normal"); // Reset to normal search when changing app mode
                setTemporalSearch({
                  events: [""],
                  results: [],
                  currentPage: 0,
                  isActive: false,
                  searchMode: "progressive"
                });
                setResults([]);
              }}
            >
              <option value="textual-kis">📋 Textual KIS</option>
              <option value="qa">❓ Q&A</option>
              <option value="trake">🎬 TRAKE</option>
            </select>
          </div>

          <div className="search-mode-type-section">
            <label className="search-mode-type-label">Search Mode Type:</label>
            <select
              className="search-mode-type-select"
              value={searchModeType}
              onChange={(e) => {
                setSearchModeType(e.target.value);
                setSelectedItems(new Set());
                setFrameAnswers({});
                if (e.target.value === "normal") {
                  setTemporalSearch({
                    events: [""],
                    results: [],
                    currentPage: 0,
                    isActive: false,
                    searchMode: "progressive"
                  });
                  setResults([]);
                } else {
                  // Reset temporal search events when switching to temporal mode
                  setTemporalSearch(prev => ({
                    ...prev,
                    events: [""],
                    results: [],
                    currentPage: 0,
                    isActive: false,
                    searchMode: "progressive"
                  }));
                  setResults([]);
                }
              }}
            >
              <option value="normal">
                🔍 Normal Search 
                {appMode === "textual-kis" ? " (Standard KIS)" :
                 appMode === "qa" ? " (Single Q&A)" :
                 " (Single Video Selection)"}
              </option>
              <option value="temporal">
                ⏰ Temporal Search
                {appMode === "textual-kis" ? " (Sequential Events)" :
                 appMode === "qa" ? " (Multi-Event Q&A)" :
                 " (Event Sequences)"}
              </option>
            </select>
          </div>

          {searchModeType === "normal" && (
          <div className="search-type-section">
            <label className="search-type-label">Search Type:</label>
            <select
              className="search-type-select"
              value={searchType}
              onChange={(e) => {
                setSearchType(e.target.value);
                if (e.target.value === "image") {
                  setQuery("");
                } else {
                  clearImage();
                }
              }}
            >
              <option value="text">📝 Text Search</option>
              <option value="image">🖼️ Image Search</option>
            </select>
          </div>
          )}

          {(searchModeType === "normal" ? searchType === "text" : true) && (
            <>
              <div className="search-mode-section">
                <label className="search-mode-label">Search Mode:</label>
                <select
                  className="search-mode-select"
                  value={searchMode}
                  onChange={(e) => setSearchMode(e.target.value)}
                >
                  <option value="hybrid">🔗 Hybrid (CLIP + Vintern)</option>
                  <option value="clip">🖼️ CLIP Only</option>
                  <option value="vintern">📝 Vintern Only</option>
                </select>
              </div>

              {(searchMode === "hybrid" || searchMode === "vintern") && (
                <div className="caption-mode-section">
                  <label className="caption-mode-label">Caption Model:</label>
                  <select
                    className="caption-mode-select"
                    value={captionMode}
                    onChange={(e) => setCaptionMode(e.target.value)}
                  >
                    <option value="bge">🇻🇳 Vietnamese_Embedding_v2</option>
                    <option value="gte">📄 vietnamese-document-embedding</option>
                  </select>
                </div>
              )}

              {searchMode === "hybrid" && (
                <div className="alpha-section">
                  <label className="alpha-label">
                    Text/Visual Balance: {(alpha * 100).toFixed(0)}% text
                    <input
                      className="alpha-slider"
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.1"
                      value={alpha}
                      onChange={(e) => setAlpha(parseFloat(e.target.value))}
                    />
                  </label>
                </div>
              )}
            </>
          )}

          {searchType === "image" && (
            <div className="image-upload-section">
              <label className="image-upload-label">Drag and Drop Image:</label>
              <div
                className="image-drop-area"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                style={{
                  border: '2px dashed #ccc',
                  padding: '20px',
                  textAlign: 'center',
                  backgroundColor: '#121212',
                  borderRadius: '8px',
                  cursor: 'pointer'
                }}
              >
                <p>Drop an image here or click to select</p>
                <input
                  id="image-upload-input"
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleImageUpload(e.target.files[0])}
                  style={{ display: 'none' }}
                />
                <button
                  onClick={() => document.getElementById('image-upload-input').click()}
                  className="select-image-btn"
                  type="button"
                >
                  Select Image
                </button>
              </div>
              {imagePreview && (
                <div className="image-preview">
                  <img
                    src={imagePreview}
                    alt="Upload preview"
                    className="preview-image"
                  />
                  <button
                    onClick={clearImage}
                    className="clear-image-btn"
                    type="button"
                  >
                    ✕ Clear
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Temporal Search Section - Only show when temporal mode is selected */}
          {searchModeType === "temporal" && (
          <div className="temporal-search-sidebar">
            <div className="temporal-header">
              <h4>⏰ Temporal Search</h4>
              <p className="temporal-description">Search sequential events</p>
            </div>


            {/* Search Mode Selection */}
            <div className="temporal-search-mode-section">
              <label className="temporal-search-mode-label">Search Strategy:</label>
              <select
                className="temporal-search-mode-select"
                value={temporalSearch.searchMode}
                onChange={(e) => setTemporalSearch(prev => ({
                  ...prev,
                  searchMode: e.target.value
                }))}
              >
                <option value="progressive">🔍 Sequential Search (Step-by-Step)</option>
                <option value="consolidated">📽️ Timeline View (Video Events)</option>
              </select>
              <div className="temporal-mode-description">
                {temporalSearch.searchMode === "progressive" ? (
                  <span style={{fontSize: "11px", color: "#666"}}>
                    Searches events step-by-step, narrowing results after each step
                  </span>
                ) : (
                  <span style={{fontSize: "11px", color: "#666"}}>
                    Shows complete video timelines with all events in sequence
                  </span>
                )}
              </div>
            </div>

            <div className="temporal-events">
              {temporalSearch.events.map((event, index) => (
                <div key={index} className="temporal-event-row">
                  <label className="event-label">E{index + 1}:</label>
                  <input
                    type="text"
                    className="temporal-event-input"
                    placeholder="Describe event..."
                    value={event}
                    onChange={(e) => handleTemporalEventChange(index, e.target.value)}
                  />
                  {temporalSearch.events.length > 1 && (
                    <button
                      className="remove-event-btn"
                      onClick={() => removeTemporalEvent(index)}
                    >
                      ✕
                    </button>
                  )}
                </div>
              ))}
            </div>

            <div className="temporal-controls">
              <button
                className="add-event-btn"
                onClick={addTemporalEvent}
              >
                + Add Event
              </button>
              <button
                className="temporal-search-btn"
                onClick={handleTemporalSearch}
                disabled={isSearching || temporalSearch.events.every(e => !e.trim())}
              >
                {isSearching ? "Searching..." : "🔍 Search"}
              </button>
              {temporalSearch.isActive && (
                <button
                  className="exit-temporal-btn"
                  onClick={exitTemporalSearch}
                >
                  ✕ Exit
                </button>
              )}
            </div>
          </div>
          )}

          {/* Video Keyframes Viewer Section */}
          <div className="video-keyframes-viewer-section">
            <div className="video-keyframes-header">
              <h4>🎬 Browse Video Keyframes</h4>
              <p className="video-keyframes-description">View all keyframes from any video</p>
            </div>

            <div className="video-selection">
              <label className="video-selection-label">Select Video:</label>
              <input
                type="text"
                className="video-selection-input"
                placeholder="Enter video ID (e.g., L01_V001)"
                value={selectedVideoForViewing}
                onChange={(e) => setSelectedVideoForViewing(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleVideoSelectionForViewing(e.target.value);
                  }
                }}
              />
              <button
                className="load-keyframes-btn"
                onClick={() => handleVideoSelectionForViewing(selectedVideoForViewing)}
                disabled={!selectedVideoForViewing || loadingVideoKeyframes}
              >
                {loadingVideoKeyframes ? "Loading..." : "🔍 Load Keyframes"}
              </button>
            </div>

            {videoKeyframes.length > 0 && (
              <div className="video-keyframes-status">
                <div className="keyframes-info">
                  <span>📊 {videoKeyframes.length} keyframes from {selectedVideoForViewing}</span>
                  <span style={{fontSize: "12px", color: "#666", marginTop: "4px", display: "block"}}>
                    View keyframes on the right side →
                  </span>
                  <button
                    className="clear-keyframes-btn"
                    onClick={() => {
                      setSelectedVideoForViewing("");
                      setVideoKeyframes([]);
                      exitVideoBrowse();
                    }}
                  >
                    ✕ Clear
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Video Player Preference Section */}
          <div className="video-player-section">
            <label className="video-player-label">Video Player:</label>
            <select
              className="video-player-select"
              value={videoPlayerPreference}
              onChange={(e) => setVideoPlayerPreference(e.target.value)}
              title="Choose how to open video results"
            >
              <option value="embedded">🌐 Embedded Player (Web)</option>
              <option value="mpc">🎬 MPC-HC/BE (External)</option>
            </select>
            <div className="video-player-description">
              {videoPlayerPreference === "embedded" 
                ? "Opens videos in built-in web player with timestamp" 
                : "Opens videos in MPC with yt-dlp support and precise timestamp"}
            </div>
          </div>

          {((results.length > 0 && appMode !== "trake") || (appMode === "trake" && trakeMode.eventSequences.length > 0) || (searchModeType === "temporal" && temporalSearch.isActive)) && (
            <div className="csv-export-section">
              <label className="csv-label">
                Export Selected ({
                  (searchModeType === "temporal" && temporalSearch.isActive) ? "Temporal Results" :
                  appMode === "qa" ? "Q&A Format" :
                    appMode === "trake" ? "TRAKE Format" :
                      "Textual KIS"
                }):
              </label>
              <input
                className="csv-filename-input"
                type="text"
                placeholder="CSV filename"
                value={csvFileName}
                onChange={(e) => setCsvFileName(e.target.value)}
              />
              <div className="csv-controls">
                <div className="selection-info">
                  {(searchModeType === "temporal" && temporalSearch.isActive) ? (
                    <>
                      {temporalSearch.results.length} temporal results
                      <div style={{ fontSize: "11px", color: "#666", marginTop: "2px" }}>
                        From {temporalSearch.events.filter(e => e.trim()).length} events
                        {appMode === "trake" && " (Final event results only)"}
                        {appMode === "qa" && " (Q&A analysis)"}
                      </div>
                    </>
                  ) : appMode === "trake" && searchModeType === "normal" ? (
                    <>
                      {trakeMode.eventSequences.length} event sequences
                      <div style={{ fontSize: "11px", color: "#666", marginTop: "2px" }}>
                        Current sequence: {trakeMode.currentSequence.length} frames
                      </div>
                    </>
                  ) : (
                    <>
                      {selectedItems.size} selected
                      {appMode === "qa" && (
                        <div style={{ fontSize: "11px", color: "#666", marginTop: "2px" }}>
                          {Object.keys(frameAnswers).filter(id => selectedItems.has(id) && frameAnswers[id]).length} with answers
                        </div>
                      )}
                    </>
                  )}
                </div>
                <div className="csv-buttons">
                  {(appMode !== "trake" || (appMode === "trake" && searchModeType === "temporal")) && (
                    <>
                      <button className="csv-btn select-all" onClick={selectAllCurrentPage}>
                        Select Page
                      </button>
                      <button className="csv-btn clear-all" onClick={clearAllSelection}>
                        Clear All
                      </button>
                    </>
                  )}
                  <button
                    className="csv-btn download"
                    onClick={downloadCSV}
                    disabled={
                      (searchModeType === "temporal" && temporalSearch.isActive) ? temporalSearch.results.length === 0 :
                      appMode === "trake" ? trakeMode.eventSequences.length === 0 : 
                      selectedItems.size === 0
                    }
                  >
                    📥 Download CSV
                  </button>
                </div>
              </div>
              {appMode === "qa" && (
                <div style={{ fontSize: "11px", color: "#666", marginTop: "5px" }}>
                  Format: videoFile,frameNum,answer
                </div>
              )}
              {appMode === "trake" && (
                <div style={{ fontSize: "11px", color: "#666", marginTop: "5px" }}>
                  Format: L10_V001, 1200, 1850, 2100, 2450
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className={`main-content ${videoBrowseMode ? 'with-video-browse' : ''}`}>
        <div className="main-search-results">
          {isSearching ? (
            <div className="empty-state" style={{ textAlign: "center", marginTop: 50 }}>
              <FaSearch size={80} color="#999" />
              <p>Đang tìm kiếm...</p>
            </div>
          ) : results.length === 0 ? (
            <div className="empty-state" style={{ textAlign: "center", marginTop: 50 }}>
              <FaSearch size={80} color="#999" />
              <p>Chưa có kết quả để hiển thị</p>
            </div>
          ) : (
          <>
            {/* Video Filter Section */}
            {currentDataSource.length > 0 && (
              <div className="video-filter-section">
                <div className="video-filter-controls">
                  <span className="video-filter-label">
                    Filter by Video:
                  </span>
                  <select
                    className="video-filter-select"
                    value={videoFilter}
                    onChange={(e) => handleVideoFilterChange(e.target.value)}
                  >
                    <option value="all">🎭 All Videos ({rawDataSource.length} frames)</option>
                    {getUniqueVideos(rawDataSource).map(videoId => {
                      const count = filterResultsByVideo(rawDataSource, videoId).length;
                      return (
                        <option key={videoId} value={videoId}>
                          📹 {videoId} ({count} frames)
                        </option>
                      );
                    })}
                  </select>
                </div>
                <div className="video-filter-info">
                  📊 Showing: {currentDataSource.length} frames
                  {videoFilter !== "all" && ` from ${videoFilter}`}
                </div>
              </div>
            )}

            {searchTime && (
              <div style={{
                textAlign: "center",
                margin: "10px 0",
                color: "#666",
                fontSize: "14px"
              }}>
                {(searchModeType === "temporal" && temporalSearch.isActive) ? (
                  <div>
                    {temporalSearch.searchMode === "consolidated" ? "🎯" : "🔄"} 
                    {appMode === "qa" ? " Q&A" : appMode === "trake" ? " TRAKE" : ""} 
                    {" "}{temporalSearch.searchMode.charAt(0).toUpperCase() + temporalSearch.searchMode.slice(1)} temporal search completed in {searchTime}s
                    <div style={{fontSize: "12px", marginTop: "2px"}}>
                      {temporalSearch.searchMode === "consolidated" ? 
                        `${results.length} video timelines from ${temporalSearch.events.filter(e => e.trim()).length} events` :
                        `${results.length} results from ${temporalSearch.events.filter(e => e.trim()).length} events`
                      }
                      {appMode === "qa" && " (optimized for Q&A)"}
                      {appMode === "trake" && temporalSearch.searchMode === "progressive" && " (final event only)"}
                      {appMode === "trake" && temporalSearch.searchMode === "consolidated" && " (video sequences)"}
                    </div>
                  </div>
                ) : (
                  `🔍 Search completed in ${searchTime}s - ${results.length} results (${searchType === "image" ? "IMAGE SEARCH" : searchMode.toUpperCase()})`
                )}
              </div>
            )}
            <div className="result-box">
              {currentPageData.map((item, idx) => (
                <div key={idx} className={`card ${selectedItems.has(item.videoId) ? 'selected' : ''} ${item.isConsolidated ? 'consolidated-card' : ''}`}>
                  <div className="card-header-new">
                    {!item.videoTimeline && (
                      <div className="regular-card-header">
                        <input
                          type="checkbox"
                          className="regular-card-checkbox"
                          checked={selectedItems.has(item.videoId)}
                          onChange={() => toggleItemSelection(item.videoId)}
                          title="Select for CSV export"
                        />
                        <span className="card-caption-new">{item.caption}</span>
                      </div>
                    )}
                  </div>
                  
                  {/* Timeline Mode: Show video timeline with consecutive events in same row (both consolidated and progressive) */}
                  {item.videoTimeline && (
                      <div className="consolidated-timeline-container">
                        <div className="timeline-header-above-frames">
                          <div className="timeline-header-content">
                            <div className="timeline-left-section">
                              <input
                                type="checkbox"
                                className="timeline-checkbox"
                                checked={selectedItems.has(item.videoId)}
                                onChange={() => toggleItemSelection(item.videoId)}
                              />
                              <span className="timeline-score-header">
                                {item.videoTimeline.video_id} | Timeline Score: {item.videoTimeline.score.toFixed(3)}
                              </span>
                            </div>
                            <button
                              className="timeline-view-frames-btn"
                              onClick={() => {
                                if (appMode === 'trake') {
                                  selectVideoForTrake(item.videoId);
                                }
                              }}
                              title="View detailed frame sequence"
                            >
                              🎬 View Frames
                            </button>
                          </div>
                        </div>
                        <div 
                          className="timeline-events-horizontal"
                          style={{'--frame-count': item.videoTimeline.frames.length}}
                        >
                          {item.videoTimeline.frames.map((frameObj, eventIdx) => (
                            <div key={eventIdx} className="timeline-event-frame">
                                <img
                                  src={(() => {
                                    const parts = frameObj.id.split('_');
                                    const batch = parts[0];
                                    const videoNumber = parts[1];
                                    const baseVideoId = `${batch}_${videoNumber}`;
                                    const batchConfig = getBatchConfig(frameObj.id);
                                    return `/${batchConfig.keyframesDir}/${batch}/${baseVideoId}/${frameObj.id}.jpg`;
                                  })()}
                                  alt={`Event ${eventIdx + 1}`}
                                  className="event-frame-image"
                                  onClick={() => handleKeyframeClick(frameObj.id)}
                                  onError={(e) => {
                                    e.target.src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBOb3QgRm91bmQ8L3RleHQ+PC9zdmc+";
                                    e.target.style.filter = "grayscale(1)";
                                  }}
                                  title={`🎥 Click to open video at timestamp ${frameObj.id}`}
                                />
                                <div className="event-frame-caption">
                                  <div className="event-number">E{eventIdx + 1}</div>
                                  <div className="event-frame-id">{frameObj.id}</div>
                                  <div className="event-frame-score">Score: {frameObj.score.toFixed(3)}</div>
                                </div>
                              {eventIdx < item.videoTimeline.frames.length - 1 && (
                                <div className="event-arrow">→</div>
                              )}
                            </div>
                          ))}
                        </div>
                        <div className="timeline-summary">
                          <div className="sequence-info">
                            🎬 Video Sequence: {item.videoTimeline.frames.length} events detected | 
                            Avg Frame Score: {(item.videoTimeline.frameScores.reduce((a, b) => a + b, 0) / item.videoTimeline.frameScores.length).toFixed(3)}
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {appMode === "qa" && selectedItems.has(item.videoId) && (
                      <div className="qa-input-section">
                        <textarea
                          className="qa-answer-input"
                          placeholder="Enter your answer for this frame..."
                          value={frameAnswers[item.videoId] || ""}
                          onChange={(e) => updateFrameAnswer(item.videoId, e.target.value)}
                          rows={2}
                        />
                      </div>
                    )}
                  
                  {/* Only show main image for non-consolidated cards */}
                  {!item.isConsolidated && (
                    <div style={{ position: 'relative' }}>
                      <img
                        src={item.image}
                        alt={`Keyframe for ${item.videoId}`}
                        onClick={() => handleKeyframeClick(item.videoId)}
                        style={{
                          cursor: loadingVideo === item.videoId ? 'wait' : 'pointer',
                          opacity: loadingVideo === item.videoId ? 0.7 : 1
                        }}
                        title={`🎥 Click to open video at timestamp ${item.videoId.split('_')[2]}`}
                        onError={(e) => {
                          // Show placeholder if keyframe not found
                          e.target.src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBOb3QgRm91bmQ8L3RleHQ+PC9zdmc+";
                          e.target.style.filter = "grayscale(1)";
                        }}
                      />
                      {loadingVideo === item.videoId && (
                        <div style={{
                          position: 'absolute',
                          top: '50%',
                          left: '50%',
                          transform: 'translate(-50%, -50%)',
                          background: 'rgba(0, 0, 0, 0.8)',
                          color: 'white',
                          padding: '8px 12px',
                          borderRadius: '6px',
                          fontSize: '12px',
                          pointerEvents: 'none'
                        }}>
                          Opening video...
                        </div>
                      )}
                      <div style={{
                        position: 'absolute',
                        top: '8px',
                        right: '8px',
                        background: 'rgba(0, 0, 0, 0.7)',
                        color: 'white',
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontSize: '10px',
                        fontWeight: 'bold'
                      }}>
                        🎥 CLICK
                      </div>
                    </div>
                  )}
                </div>
              ))}
              {Array.from({ length: missingCount }).map((_, idx) => (
                <div key={"empty-" + idx} className="card-empty"></div>
              ))}
            </div>
            <div className="pagination">
              <button onClick={() => goToPage(((searchModeType === "temporal" && temporalSearch.isActive) ? temporalSearch.currentPage : pageIndex) - 1)} disabled={((searchModeType === "temporal" && temporalSearch.isActive) ? temporalSearch.currentPage : pageIndex) === 0}>
                Previous
              </button>

              <span>
                Page {((searchModeType === "temporal" && temporalSearch.isActive) ? temporalSearch.currentPage : pageIndex) + 1} of {totalPages}
                {(searchModeType === "temporal" && temporalSearch.isActive) && (
                  <div style={{ fontSize: "11px", color: "#666", marginTop: "2px" }}>
                    Temporal Search Results ({temporalFrameSize} per page)
                  </div>
                )}
              </span>

              <button
                onClick={() => goToPage(((searchModeType === "temporal" && temporalSearch.isActive) ? temporalSearch.currentPage : pageIndex) + 1)}
                disabled={((searchModeType === "temporal" && temporalSearch.isActive) ? temporalSearch.currentPage : pageIndex) === totalPages - 1}
              >
                Next
              </button>
            </div>
          </>
        )}

        {/* TRAKE Mode Video Frames Viewer */}
        {appMode === "trake" && trakeMode.selectedVideo && trakeMode.videoFrames.length > 0 && (
          <div className="trake-video-section">
            <div className="trake-header">
              <h3>🎬 Video Frames: {trakeMode.selectedVideo}</h3>
              <div className="trake-controls">
                <div className="sequence-info">
                  Current Sequence: [{trakeMode.currentSequence.join(', ')}]
                </div>
                <button
                  className="save-sequence-btn"
                  onClick={saveEventSequence}
                  disabled={trakeMode.currentSequence.length === 0}
                >
                  💾 Save Sequence
                </button>
              </div>
            </div>

            <div className="trake-pagination-info">
              Showing {trakeMode.currentPage * 10 + 1}-{Math.min((trakeMode.currentPage + 1) * 10, trakeMode.videoFrames.length)} of {trakeMode.videoFrames.length} frames
            </div>

            <div className="trake-frames-grid">
              {trakeMode.videoFrames
                .slice(trakeMode.currentPage * 10, (trakeMode.currentPage + 1) * 10)
                .map((frame, idx) => (
                  <div key={frame.videoId} className={`trake-frame ${trakeMode.currentSequence.includes(frame.frameNum) ? 'selected' : ''}`}>
                    <div className="trake-frame-controls">
                      <input
                        type="checkbox"
                        checked={trakeMode.currentSequence.includes(frame.frameNum)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            addFrameToSequence(frame.frameNum);
                          } else {
                            removeFrameFromSequence(frame.frameNum);
                          }
                        }}
                      />
                      <span className="frame-number">Frame {frame.frameNum}</span>
                    </div>
                    <div className="trake-frame-image-container">
                      <img
                        src={frame.imagePath}
                        alt={`Frame ${frame.frameNum}`}
                        className="trake-frame-image"
                        onClick={() => setZoomedFrame(frame)}
                        onError={(e) => {
                          e.target.src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5GcmFtZSBOb3QgRm91bmQ8L3RleHQ+PC9zdmc+";
                          e.target.style.filter = "grayscale(1)";
                        }}
                      />
                      <div className="zoom-indicator">🔍</div>
                    </div>
                  </div>
                ))}
            </div>

            {trakeMode.videoFrames.length > 10 && (
              <div className="trake-pagination">
                <button
                  onClick={() => goToTrakePage(trakeMode.currentPage - 1)}
                  disabled={trakeMode.currentPage === 0}
                  className="trake-page-btn"
                >
                  Previous
                </button>

                <span className="trake-page-info">
                  Page {trakeMode.currentPage + 1} of {Math.ceil(trakeMode.videoFrames.length / 10)}
                </span>

                <button
                  onClick={() => goToTrakePage(trakeMode.currentPage + 1)}
                  disabled={trakeMode.currentPage >= Math.ceil(trakeMode.videoFrames.length / 10) - 1}
                  className="trake-page-btn"
                >
                  Next
                </button>
              </div>
            )}

            {trakeMode.eventSequences.length > 0 && (
              <div className="saved-sequences">
                <h4>💾 Saved Event Sequences</h4>
                {trakeMode.eventSequences.map((sequence, idx) => (
                  <div key={idx} className="saved-sequence">
                    <span className="sequence-text">
                      {sequence.videoId}: [{sequence.frames.join(', ')}]
                    </span>
                    <button
                      className="delete-sequence-btn"
                      onClick={() => deleteEventSequence(idx)}
                    >
                      🗑️ Delete
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        </div>

        {/* Video Browse Display (Right Side) */}
        {videoBrowseMode && videoKeyframes.length > 0 && (
          <div className="video-browse-section">
            <div className="video-browse-header">
              <h3>🎬 Browse Video: {browsedVideoId}</h3>
              <div className="video-browse-controls">
                <span className="browse-info">
                  {videoKeyframes.length} frames total
                </span>
                <button
                  className="exit-browse-btn"
                  onClick={exitVideoBrowse}
                >
                  ✕ Close Browse
                </button>
              </div>
            </div>

            <div className="video-browse-content">
              {(() => {
                const browseFramesPerPage = 8;
                const totalBrowsePages = Math.ceil(videoKeyframes.length / browseFramesPerPage);
                const currentBrowseFrames = videoKeyframes.slice(
                  browseCurrentPage * browseFramesPerPage,
                  (browseCurrentPage + 1) * browseFramesPerPage
                );

                return (
                  <>
                    <div className="browse-frames-grid">
                      {currentBrowseFrames.map((frame, index) => (
                        <div key={index} className="browse-frame-item">
                          <img
                            src={frame.image}
                            alt={`Frame ${frame.timestamp}`}
                            className="browse-frame-image"
                            onClick={() => setBrowseZoomedFrame(frame)}
                            onError={(e) => {
                              e.target.src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5GcmFtZSBOb3QgRm91bmQ8L3RleHQ+PC9zdmc+";
                              e.target.style.filter = "grayscale(1)";
                            }}
                          />
                          <div className="browse-frame-info">
                            <div className="browse-frame-filename">{frame.videoId}</div>
                            <div className="browse-zoom-hint">🔍 Click to zoom</div>
                          </div>
                          <div
                            className="browse-play-btn"
                            onClick={() => handleKeyframeClick(frame.videoId)}
                            title="Play video at this frame"
                          >
                            ▶️
                          </div>
                        </div>
                      ))}
                    </div>

                    {totalBrowsePages > 1 && (
                      <div className="browse-pagination">
                        <button
                          onClick={() => goToBrowsePage(browseCurrentPage - 1)}
                          disabled={browseCurrentPage === 0}
                          className="browse-page-btn"
                        >
                          ← Previous
                        </button>

                        <span className="browse-page-info">
                          Page {browseCurrentPage + 1} of {totalBrowsePages}
                          <div style={{ fontSize: "12px", color: "#666", marginTop: "2px" }}>
                            Showing {currentBrowseFrames.length} of {videoKeyframes.length} frames
                          </div>
                        </span>

                        <button
                          onClick={() => goToBrowsePage(browseCurrentPage + 1)}
                          disabled={browseCurrentPage >= totalBrowsePages - 1}
                          className="browse-page-btn"
                        >
                          Next →
                        </button>
                      </div>
                    )}
                  </>
                );
              })()}
            </div>
          </div>
        )}

        {/* Embedded Video Player */}
        {videoPlayer && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <div style={{
              position: 'relative',
              width: '90%',
              maxWidth: '800px',
              backgroundColor: '#000',
              borderRadius: '8px',
              padding: '20px'
            }}>
              {/* Close button */}
              <button
                onClick={() => setVideoPlayer(null)}
                style={{
                  position: 'absolute',
                  top: '10px',
                  right: '10px',
                  background: '#ff4444',
                  color: 'white',
                  border: 'none',
                  borderRadius: '50%',
                  width: '40px',
                  height: '40px',
                  fontSize: '20px',
                  cursor: 'pointer',
                  zIndex: 1001
                }}
              >
                ×
              </button>

              {/* Video title and keyframe info */}
              <div style={{
                color: 'white',
                marginBottom: '10px',
                fontSize: '16px',
                fontWeight: 'bold'
              }}>
                {videoPlayer.title}
              </div>
              <div style={{
                color: '#ccc',
                marginBottom: '15px',
                fontSize: '14px'
              }}>
                Initial Keyframe: {videoPlayer.keyframe} → {Math.floor(videoPlayer.timestamp / 60)}:{String(videoPlayer.timestamp % 60).padStart(2, '0')}
                {videoPlayer.fps && (
                  <div style={{ fontSize: '12px', marginTop: '4px' }}>
                    FPS: {videoPlayer.fps} | Initial Frame: {videoPlayer.keyframe.split('_')[2]} | Duration: {Math.floor(videoPlayer.videoDuration / 60)}:{String(videoPlayer.videoDuration % 60).padStart(2, '0')}
                  </div>
                )}
              </div>

              {/* Simple Time to Frame Calculator */}
              <div style={{
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                border: '1px solid rgba(0, 123, 255, 0.3)',
                borderRadius: '4px',
                padding: '10px',
                marginBottom: '10px'
              }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginBottom: '8px'
                }}>
                  <span style={{ color: '#007bff', fontSize: '12px', fontWeight: 'bold', minWidth: '70px' }}>
                    Enter Time:
                  </span>
                  <input
                    type="text"
                    placeholder="00:56.497 or 56.497 (with milliseconds)"
                    value={timeInput}
                    onChange={(e) => setTimeInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        handleTimeToFrameConversion();
                      }
                    }}
                    style={{
                      padding: '4px 8px',
                      borderRadius: '3px',
                      border: '1px solid #555',
                      backgroundColor: '#2a2a2a',
                      color: 'white',
                      fontSize: '12px',
                      flex: 1
                    }}
                  />
                  <button
                    onClick={handleTimeToFrameConversion}
                    style={{
                      padding: '4px 8px',
                      borderRadius: '3px',
                      border: 'none',
                      backgroundColor: '#007bff',
                      color: 'white',
                      fontSize: '11px',
                      cursor: 'pointer'
                    }}
                  >
                    ➡️ Convert
                  </button>
                </div>

                {calculatedFrameOrder && (
                  <div style={{
                    backgroundColor: 'rgba(0, 255, 0, 0.1)',
                    border: '1px solid rgba(0, 255, 0, 0.3)',
                    borderRadius: '3px',
                    padding: '6px',
                    textAlign: 'center'
                  }}>
                    <div style={{
                      fontSize: '14px',
                      fontWeight: 'bold',
                      color: '#00ff00',
                      marginBottom: '2px'
                    }}>
                      Frame Order: {calculatedFrameOrder.frameOrder.toLocaleString()}
                    </div>
                    <div style={{
                      fontSize: '10px',
                      color: '#888',
                      fontFamily: 'monospace'
                    }}>
                      {calculatedFrameOrder.preciseCalculation}
                    </div>
                  </div>
                )}

                <div style={{
                  fontSize: '9px',
                  color: '#888',
                  textAlign: 'center',
                  marginTop: '4px'
                }}>
                  Video FPS: {videoPlayer.fps} | Duration: {Math.floor(videoPlayer.videoDuration / 60)}:{String(videoPlayer.videoDuration % 60).padStart(2, '0')}
                </div>
              </div>


              {/* YouTube iframe */}
              <iframe
                key={videoPlayer.timestamp}
                width="100%"
                height="450"
                src={`https://www.youtube.com/embed/${videoPlayer.videoId}?start=${videoPlayer.timestamp}&autoplay=1&rel=0&modestbranding=1`}
                title={videoPlayer.title}
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                style={{
                  borderRadius: '8px'
                }}
              ></iframe>

              {/* Instructions */}
              <div style={{
                color: '#999',
                marginTop: '10px',
                fontSize: '12px',
                textAlign: 'center'
              }}>
                Video will start at the keyframe timestamp. Press ESC or click × to close.
              </div>
            </div>
          </div>
        )}

        {/* Frame Zoom Modal for Browse */}
        {browseZoomedFrame && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.95)',
            zIndex: 1001,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px'
          }}>
            <div style={{
              position: 'relative',
              maxWidth: '95vw',
              maxHeight: '95vh',
              backgroundColor: '#1f1f1f',
              borderRadius: '12px',
              padding: '20px',
              overflow: 'hidden'
            }}>
              {/* Close button */}
              <button
                onClick={() => setBrowseZoomedFrame(null)}
                style={{
                  position: 'absolute',
                  top: '15px',
                  right: '15px',
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '50%',
                  width: '40px',
                  height: '40px',
                  fontSize: '18px',
                  cursor: 'pointer',
                  zIndex: 1002,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                ×
              </button>

              {/* Frame info */}
              <div style={{
                color: 'white',
                marginBottom: '15px',
                fontSize: '18px',
                fontWeight: 'bold',
                textAlign: 'center'
              }}>
                {browsedVideoId} - {browseZoomedFrame.videoId}
              </div>

              {/* Zoomed image */}
              <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                maxWidth: 'calc(95vw - 40px)',
                maxHeight: 'calc(95vh - 120px)',
                overflow: 'auto'
              }}>
                <img
                  src={browseZoomedFrame.image}
                  alt={`Frame ${browseZoomedFrame.timestamp} - Zoomed`}
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain',
                    borderRadius: '8px',
                    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)'
                  }}
                  onError={(e) => {
                    e.target.src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMTgiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5GcmFtZSBOb3QgRm91bmQ8L3RleHQ+PC9zdmc+";
                  }}
                />
              </div>

              {/* Play video button */}
              <div style={{
                textAlign: 'center',
                marginTop: '15px'
              }}>
                <button
                  onClick={() => {
                    setBrowseZoomedFrame(null);
                    handleKeyframeClick(browseZoomedFrame.videoId);
                  }}
                  style={{
                    backgroundColor: '#007bff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    padding: '8px 16px',
                    fontSize: '14px',
                    cursor: 'pointer',
                    marginRight: '10px'
                  }}
                >
                  ▶️ Play Video ({browseZoomedFrame.videoId})
                </button>
              </div>

              {/* Instructions */}
              <div style={{
                color: '#999',
                marginTop: '15px',
                fontSize: '12px',
                textAlign: 'center'
              }}>
                Press ESC or click × to close zoom view
              </div>
            </div>
          </div>
        )}

        {/* Frame Zoom Modal for TRAKE */}
        {zoomedFrame && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.95)',
            zIndex: 1001,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px'
          }}>
            <div style={{
              position: 'relative',
              maxWidth: '95vw',
              maxHeight: '95vh',
              backgroundColor: '#1f1f1f',
              borderRadius: '12px',
              padding: '20px',
              overflow: 'hidden'
            }}>
              {/* Close button */}
              <button
                onClick={() => setZoomedFrame(null)}
                style={{
                  position: 'absolute',
                  top: '15px',
                  right: '15px',
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '50%',
                  width: '40px',
                  height: '40px',
                  fontSize: '18px',
                  cursor: 'pointer',
                  zIndex: 1002,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                ×
              </button>

              {/* Frame info */}
              <div style={{
                color: 'white',
                marginBottom: '15px',
                fontSize: '18px',
                fontWeight: 'bold',
                textAlign: 'center'
              }}>
                {trakeMode.selectedVideo} - Frame {zoomedFrame.frameNum}
              </div>

              {/* Zoomed image */}
              <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                maxWidth: 'calc(95vw - 40px)',
                maxHeight: 'calc(95vh - 120px)',
                overflow: 'auto'
              }}>
                <img
                  src={zoomedFrame.imagePath}
                  alt={`Frame ${zoomedFrame.frameNum} - Zoomed`}
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain',
                    borderRadius: '8px',
                    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)'
                  }}
                  onError={(e) => {
                    e.target.src = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzMzIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMTgiIGZpbGw9IiM5OTkiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5GcmFtZSBOb3QgRm91bmQ8L3RleHQ+PC9zdmc+";
                  }}
                />
              </div>

              {/* Instructions */}
              <div style={{
                color: '#999',
                marginTop: '15px',
                fontSize: '12px',
                textAlign: 'center'
              }}>
                Click image to view full resolution. Press ESC or click × to close.
              </div>
            </div>
          </div>
        )}

        <div className="search-row">
          {searchModeType === "temporal" ? (
            // Temporal search interface
            <div className="temporal-search-interface">
              <div className="temporal-search-info">
                <span>
                  {appMode === "textual-kis" && "🔍 Sequential Event Search"}
                  {appMode === "qa" && "❓ Multi-Event Q&A Analysis"} 
                  {appMode === "trake" && "🎬 Video Sequence Analysis"}
                </span>
                <div style={{fontSize: "12px", color: "#888", marginTop: "4px"}}>
                  Use the temporal search controls in the sidebar to define events
                </div>
              </div>
            </div>
          ) : searchType === "text" ? (
            // Normal text search
            <input
              type="text"
              placeholder={
                appMode === "textual-kis" ? "Enter search query for KIS..." :
                appMode === "qa" ? "Enter question or topic..." :
                "Enter search query for video selection..."
              }
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  search();
                }
              }}
            />
          ) : (
            // Image search interface
            <div className="image-search-display">
              {imagePreview ? (
                <div className="search-image-preview">
                  <img src={imagePreview} alt="Search query" />
                  <span>Image uploaded - Click search to find similar images</span>
                </div>
              ) : (
                <div className="no-image-placeholder">
                  <span>No image selected - Drag and drop an image in settings</span>
                </div>
              )}
            </div>
          )}
          <button 
            id="search-btn" 
            onClick={search} 
            disabled={
              isSearching || 
              (searchModeType === "temporal" && temporalSearch.events.every(e => !e.trim()))
            }
            style={{
              display: searchModeType === "temporal" ? "none" : "block"
            }}
          >
            {isSearching ? "..." : <FaArrowUp />}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;