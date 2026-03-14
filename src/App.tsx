/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Upload, 
  Settings, 
  Image as ImageIcon, 
  Video, 
  Pipette, 
  Trash2, 
  Download, 
  Play, 
  Pause,
  Layers,
  Sparkles,
  RefreshCcw,
  Pencil,
  Hexagon,
  Eraser,
  Undo2,
  Maximize2,
  MousePointer2,
  Wand2,
  Loader2
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { SelfieSegmentation } from '@mediapipe/selfie_segmentation';
import { GoogleGenAI, Type } from "@google/genai";
import { cn } from './lib/utils';

// --- Types ---

type RemovalMode = 'chroma' | 'ai' | 'mask';
type MaskTool = 'brush' | 'polygon' | 'eraser' | 'none';

interface Point {
  x: number;
  y: number;
}

interface ProcessingSettings {
  mode: RemovalMode;
  chromaColor: { r: number; g: number; b: number };
  similarity: number;
  smoothness: number;
  spill: number;
  backgroundType: 'transparent' | 'color' | 'image';
  backgroundColor: string;
  backgroundImage: string | null;
  maskTool: MaskTool;
  brushSize: number;
  maskFeather: number;
  invertMask: boolean;
}

// --- Constants ---

const DEFAULT_SETTINGS: ProcessingSettings = {
  mode: 'chroma',
  chromaColor: { r: 0, g: 255, b: 0 }, // Default Green
  similarity: 0.15,
  smoothness: 0.08,
  spill: 0.1,
  backgroundType: 'transparent',
  backgroundColor: '#000000',
  backgroundImage: null,
  maskTool: 'brush',
  brushSize: 40,
  maskFeather: 10,
  invertMask: false,
};

// --- Main Component ---

export default function App() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [settings, setSettings] = useState<ProcessingSettings>(DEFAULT_SETTINGS);
  const [isAiModelLoading, setIsAiModelLoading] = useState(false);
  const [isRefining, setIsRefining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Masking State
  const [isDrawing, setIsDrawing] = useState(false);
  const [polygonPoints, setPolygonPoints] = useState<Point[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bgImageRef = useRef<HTMLImageElement>(null);
  const requestRef = useRef<number | null>(null);
  const segmentationRef = useRef<SelfieSegmentation | null>(null);

  // Initialize Mask Canvas
  useEffect(() => {
    if (!maskCanvasRef.current) {
      maskCanvasRef.current = document.createElement('canvas');
    }
  }, []);

  // --- AI Model Initialization ---

  useEffect(() => {
    if (settings.mode === 'ai' && !segmentationRef.current) {
      setIsAiModelLoading(true);
      const segmentation = new SelfieSegmentation({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`,
      });

      segmentation.setOptions({
        modelSelection: 1,
      });

      segmentation.onResults((results) => {
        if (!canvasRef.current || !videoRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw background first
        drawBackground(ctx, canvas.width, canvas.height);

        // Draw the segmentation mask
        ctx.globalCompositeOperation = 'destination-atop';
        ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
        
        ctx.globalCompositeOperation = 'destination-in';
        ctx.drawImage(results.segmentationMask, 0, 0, canvas.width, canvas.height);
        
        // Apply manual mask if in mask mode or if mask exists
        applyManualMask(ctx, canvas.width, canvas.height);
        
        ctx.restore();
      });

      segmentationRef.current = segmentation;
      setIsAiModelLoading(false);
    }
  }, [settings.mode]);

  // --- Processing Loop ---

  const drawBackground = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    if (settings.backgroundType === 'color') {
      ctx.fillStyle = settings.backgroundColor;
      ctx.fillRect(0, 0, width, height);
    } else if (settings.backgroundType === 'image' && settings.backgroundImage && bgImageRef.current) {
      ctx.drawImage(bgImageRef.current, 0, 0, width, height);
    }
  };

  const applyManualMask = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    if (!maskCanvasRef.current) return;
    
    // If we are in mask mode or have a mask, we might want to use it
    // For simplicity, we'll always composite the mask if we're in mask mode
    if (settings.mode === 'mask') {
      ctx.save();
      ctx.globalCompositeOperation = settings.invertMask ? 'destination-out' : 'destination-in';
      ctx.filter = `blur(${settings.maskFeather}px)`;
      ctx.drawImage(maskCanvasRef.current, 0, 0, width, height);
      ctx.restore();
    }
  };

  const processFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || videoRef.current.paused || videoRef.current.ended) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    if (!ctx) return;

    if (settings.mode === 'ai') {
      if (segmentationRef.current) {
        await segmentationRef.current.send({ image: video });
      }
    } else if (settings.mode === 'chroma') {
      // Chroma Key Processing
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const length = frame.data.length;

      const { r: targetR, g: targetG, b: targetB } = settings.chromaColor;
      const similarity = settings.similarity * 255;
      const smoothness = settings.smoothness * 255;
      const spill = settings.spill;

      for (let i = 0; i < length; i += 4) {
        const r = frame.data[i];
        const g = frame.data[i + 1];
        const b = frame.data[i + 2];

        const diffR = r - targetR;
        const diffG = g - targetG;
        const diffB = b - targetB;
        const distance = Math.sqrt(diffR * diffR + diffG * diffG + diffB * diffB);

        let alpha = 255;
        if (distance < similarity) {
          alpha = 0;
        } else if (distance < similarity + smoothness) {
          alpha = ((distance - similarity) / smoothness) * 255;
        }

        frame.data[i + 3] = alpha;

        if (alpha < 255) {
          const avg = (r + b) / 2;
          if (g > avg) {
            frame.data[i + 1] = g - (g - avg) * spill;
          }
        }
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawBackground(ctx, canvas.width, canvas.height);
      
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      if (tempCtx) {
        tempCtx.putImageData(frame, 0, 0);
        ctx.drawImage(tempCanvas, 0, 0);
      }
    } else if (settings.mode === 'mask') {
      // Manual Masking Mode
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawBackground(ctx, canvas.width, canvas.height);
      
      ctx.save();
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      applyManualMask(ctx, canvas.width, canvas.height);
      ctx.restore();
    }

    requestRef.current = requestAnimationFrame(processFrame);
  }, [settings]);

  useEffect(() => {
    if (isPlaying) {
      requestRef.current = requestAnimationFrame(processFrame);
    } else if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isPlaying, processFrame]);

  // --- Masking Handlers ---

  const getMousePos = (e: React.MouseEvent | React.TouchEvent) => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    let clientX, clientY;
    if ('touches' in e) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = (e as React.MouseEvent).clientX;
      clientY = (e as React.MouseEvent).clientY;
    }

    return {
      x: ((clientX - rect.left) / rect.width) * canvas.width,
      y: ((clientY - rect.top) / rect.height) * canvas.height
    };
  };

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    if (settings.mode !== 'mask') return;
    
    const pos = getMousePos(e);
    
    if (settings.maskTool === 'polygon') {
      setPolygonPoints(prev => [...prev, pos]);
      drawPolygonPreview([...polygonPoints, pos]);
    } else if (settings.maskTool === 'brush' || settings.maskTool === 'eraser') {
      setIsDrawing(true);
      drawOnMask(pos, true);
    }
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing || settings.mode !== 'mask') return;
    const pos = getMousePos(e);
    drawOnMask(pos, false);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    if (!isPlaying) processFrame(); // Refresh preview
  };

  const drawOnMask = (pos: Point, isStart: boolean) => {
    if (!maskCanvasRef.current) return;
    const ctx = maskCanvasRef.current.getContext('2d');
    if (!ctx) return;

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = settings.brushSize;
    ctx.globalCompositeOperation = settings.maskTool === 'eraser' ? 'destination-out' : 'source-over';
    ctx.strokeStyle = 'white';
    ctx.fillStyle = 'white';

    if (isStart) {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, settings.brushSize / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(pos.x, pos.y);
    } else {
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();
    }
  };

  const drawPolygonPreview = (points: Point[]) => {
    if (!canvasRef.current) return;
    // We don't draw directly on mask yet, just preview on main canvas if needed
    // But for simplicity, we'll just handle the final fill
  };

  const finishPolygon = () => {
    if (polygonPoints.length < 3 || !maskCanvasRef.current) return;
    const ctx = maskCanvasRef.current.getContext('2d');
    if (!ctx) return;

    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
    polygonPoints.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.closePath();
    ctx.fill();
    
    setPolygonPoints([]);
    if (!isPlaying) processFrame();
  };

  const clearMask = () => {
    if (!maskCanvasRef.current) return;
    const ctx = maskCanvasRef.current.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);
    setPolygonPoints([]);
    if (!isPlaying) processFrame();
  };

  const refineMaskWithAI = async () => {
    if (!canvasRef.current || !maskCanvasRef.current || !videoRef.current) return;
    
    setIsRefining(true);
    setError(null);

    try {
      const canvas = canvasRef.current;
      const maskCanvas = maskCanvasRef.current;
      
      // 1. Capture current frame from the video
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = canvas.width;
      tempCanvas.height = canvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      if (!tempCtx) throw new Error("Could not create temp context");
      
      tempCtx.drawImage(videoRef.current, 0, 0, tempCanvas.width, tempCanvas.height);
      const frameData = tempCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
      
      // 2. Capture current mask
      const maskData = maskCanvas.toDataURL('image/png').split(',')[1];

      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
      
      const prompt = `
        I have a video frame and a rough manual mask. 
        The first image is the video frame. 
        The second image is the manual mask (white is the selected area).
        Please detect the primary object that the user is trying to select with this rough mask.
        Provide a precise, high-fidelity polygon that tightly follows the object's boundaries.
        Return the result as a JSON object with a 'points' property, which is an array of {x, y} coordinates.
        Coordinates should be normalized from 0 to 1000 (0,0 is top-left, 1000,1000 is bottom-right).
        Be extremely precise with the object edges.
      `;

      const result = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: [
          {
            parts: [
              { text: prompt },
              { inlineData: { mimeType: "image/jpeg", data: frameData } },
              { inlineData: { mimeType: "image/png", data: maskData } }
            ]
          }
        ],
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              points: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    x: { type: Type.NUMBER },
                    y: { type: Type.NUMBER }
                  },
                  required: ["x", "y"]
                }
              }
            },
            required: ["points"]
          }
        }
      });

      const text = result.text;
      if (!text) throw new Error("No response from AI");
      
      const response = JSON.parse(text);
      if (response.points && response.points.length > 0) {
        // Draw the refined polygon on the mask canvas
        const ctx = maskCanvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
          ctx.fillStyle = 'white';
          ctx.beginPath();
          response.points.forEach((p: any, i: number) => {
            const x = (p.x / 1000) * maskCanvas.width;
            const y = (p.y / 1000) * maskCanvas.height;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.closePath();
          ctx.fill();
          
          if (!isPlaying) processFrame();
        }
      } else {
        setError("AI could not find a clear object. Try a more precise rough mask.");
      }
    } catch (err) {
      console.error("AI Refinement Error:", err);
      setError("AI failed to detect object. Please check your connection and try again.");
    } finally {
      setIsRefining(false);
    }
  };

  // --- General Handlers ---

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setIsPlaying(false);
      setError(null);
      setPolygonPoints([]);
      
      // Reset mask canvas size
      const video = document.createElement('video');
      video.src = URL.createObjectURL(file);
      video.onloadedmetadata = () => {
        if (maskCanvasRef.current) {
          maskCanvasRef.current.width = video.videoWidth;
          maskCanvasRef.current.height = video.videoHeight;
        }
      };
    }
  };

  const handleBgImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setSettings(prev => ({ ...prev, backgroundImage: url, backgroundType: 'image' }));
    }
  };

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play().catch(err => {
          console.error("Playback failed:", err);
          setError("Video playback failed. Try another format.");
        });
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (settings.mode === 'chroma') {
      const pos = getMousePos(e);
      const ctx = canvasRef.current?.getContext('2d');
      if (ctx) {
        const pixel = ctx.getImageData(pos.x, pos.y, 1, 1).data;
        setSettings(prev => ({
          ...prev,
          chromaColor: { r: pixel[0], g: pixel[1], b: pixel[2] }
        }));
      }
    } else if (settings.mode === 'mask') {
      startDrawing(e);
    }
  };

  const handleDownload = () => {
    if (!canvasRef.current) return;
    const link = document.createElement('a');
    link.download = 'processed-frame.png';
    link.href = canvasRef.current.toDataURL('image/png');
    link.click();
  };

  // --- Render Helpers ---

  const ControlGroup = ({ title, children, icon: Icon }: { title: string; children: React.ReactNode; icon?: any }) => (
    <div className="space-y-3 mb-6">
      <h3 className="text-xs font-bold uppercase tracking-widest text-zinc-500 flex items-center gap-2">
        {Icon ? <Icon className="w-3 h-3" /> : <Settings className="w-3 h-3" />}
        {title}
      </h3>
      <div className="space-y-4">
        {children}
      </div>
    </div>
  );

  const Slider = ({ label, value, min, max, step, onChange }: any) => (
    <div className="space-y-2">
      <div className="flex justify-between text-[10px] font-mono text-zinc-400">
        <span>{label}</span>
        <span>{value.toFixed(2)}</span>
      </div>
      <input 
        type="range" 
        min={min} 
        max={max} 
        step={step} 
        value={value} 
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
      />
    </div>
  );

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-zinc-100 font-sans selection:bg-emerald-500/30">
      {/* Header */}
      <header className="border-b border-zinc-800/50 bg-zinc-900/30 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Sparkles className="w-5 h-5 text-black" />
            </div>
            <h1 className="text-lg font-bold tracking-tight">VisionKey</h1>
          </div>
          
          <div className="flex items-center gap-4">
            {videoFile && (
              <button 
                onClick={handleDownload}
                className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-full text-sm font-medium transition-all"
              >
                <Download className="w-4 h-4" />
                Export Frame
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Preview */}
        <div className="lg:col-span-8 space-y-6">
          <div className="relative aspect-video bg-zinc-900 rounded-2xl overflow-hidden border border-zinc-800 group">
            {!videoUrl ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-12 text-center">
                <div className="w-20 h-20 bg-zinc-800 rounded-3xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-500">
                  <Video className="w-10 h-10 text-zinc-500" />
                </div>
                <h2 className="text-xl font-bold mb-2">Import your video</h2>
                <p className="text-zinc-500 max-w-xs mb-8">
                  Upload a video to begin background removal or manual masking.
                </p>
                <label className="px-8 py-3 bg-emerald-500 hover:bg-emerald-400 text-black font-bold rounded-full cursor-pointer transition-all shadow-xl shadow-emerald-500/20 active:scale-95">
                  Choose Video
                  <input type="file" accept="video/*" className="hidden" onChange={handleFileUpload} />
                </label>
              </div>
            ) : (
              <>
                <video 
                  ref={videoRef}
                  src={videoUrl}
                  className="hidden"
                  loop
                  muted
                  playsInline
                  onLoadedMetadata={() => {
                    if (canvasRef.current && videoRef.current) {
                      canvasRef.current.width = videoRef.current.videoWidth;
                      canvasRef.current.height = videoRef.current.videoHeight;
                      if (maskCanvasRef.current) {
                        maskCanvasRef.current.width = videoRef.current.videoWidth;
                        maskCanvasRef.current.height = videoRef.current.videoHeight;
                      }
                    }
                  }}
                />
                
                <canvas 
                  ref={canvasRef}
                  onMouseDown={handleCanvasClick}
                  onMouseMove={draw}
                  onMouseUp={stopDrawing}
                  onMouseLeave={stopDrawing}
                  onTouchStart={handleCanvasClick}
                  onTouchMove={draw}
                  onTouchEnd={stopDrawing}
                  className={cn(
                    "w-full h-full object-contain cursor-crosshair",
                    settings.backgroundType === 'transparent' && "bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')]"
                  )}
                />

                {/* Polygon Preview Overlay */}
                {polygonPoints.length > 0 && (
                  <svg className="absolute inset-0 pointer-events-none w-full h-full">
                    <polyline
                      points={polygonPoints.map(p => {
                        const canvas = canvasRef.current;
                        if (!canvas) return "0,0";
                        const rect = canvas.getBoundingClientRect();
                        const x = (p.x / canvas.width) * rect.width;
                        const y = (p.y / canvas.height) * rect.height;
                        return `${x},${y}`;
                      }).join(' ')}
                      fill="rgba(16, 185, 129, 0.2)"
                      stroke="#10b981"
                      strokeWidth="2"
                    />
                    {polygonPoints.map((p, i) => {
                      const canvas = canvasRef.current;
                      if (!canvas) return null;
                      const rect = canvas.getBoundingClientRect();
                      const x = (p.x / canvas.width) * rect.width;
                      const y = (p.y / canvas.height) * rect.height;
                      return <circle key={i} cx={x} cy={y} r="4" fill="#10b981" />;
                    })}
                  </svg>
                )}

                {/* Controls Overlay */}
                <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-4 px-6 py-3 bg-black/60 backdrop-blur-xl border border-white/10 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  <button onClick={togglePlay} className="p-2 hover:bg-white/10 rounded-full transition-colors">
                    {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 fill-current" />}
                  </button>
                  <div className="w-px h-4 bg-white/20" />
                  <span className="text-xs font-mono text-zinc-400">
                    {videoRef.current?.videoWidth}x{videoRef.current?.videoHeight}
                  </span>
                  <button 
                    onClick={() => {
                      if (videoRef.current) videoRef.current.currentTime = 0;
                    }}
                    className="p-2 hover:bg-white/10 rounded-full transition-colors"
                  >
                    <RefreshCcw className="w-4 h-4" />
                  </button>
                </div>

                {isAiModelLoading && (
                  <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center">
                    <div className="w-12 h-12 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin mb-4" />
                    <p className="text-sm font-medium text-emerald-500">Loading AI Model...</p>
                  </div>
                )}
              </>
            )}
          </div>

          {settings.backgroundImage && (
            <img 
              ref={bgImageRef} 
              src={settings.backgroundImage} 
              className="hidden" 
              alt="background" 
              onLoad={() => {
                if (isPlaying) processFrame();
              }}
            />
          )}

          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm flex items-center gap-3">
              <Trash2 className="w-4 h-4" />
              {error}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-6 bg-zinc-900/50 border border-zinc-800 rounded-2xl">
              <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
                <Pencil className="w-4 h-4 text-emerald-500" />
                Masking Tips
              </h3>
              <ul className="text-xs text-zinc-400 space-y-3 list-disc list-inside">
                <li>Use **Brush** for freehand isolation.</li>
                <li>Use **Polygon** for precise geometric shapes. Click points and press "Finish" to fill.</li>
                <li>Adjust **Feathering** for softer edges.</li>
                <li>**Invert Mask** to switch between isolating and removing areas.</li>
              </ul>
            </div>
            <div className="p-6 bg-zinc-900/50 border border-zinc-800 rounded-2xl">
              <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
                <Layers className="w-4 h-4 text-emerald-500" />
                Export Options
              </h3>
              <p className="text-xs text-zinc-400 mb-4">
                Export high-quality snapshots of processed frames. Full video export coming soon.
              </p>
              <button 
                disabled={!videoUrl}
                onClick={handleDownload}
                className="w-full py-2 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-xs font-bold transition-colors"
              >
                Capture Current Frame
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Controls */}
        <div className="lg:col-span-4 space-y-6">
          <div className="p-8 bg-zinc-900 border border-zinc-800 rounded-3xl sticky top-24 max-h-[calc(100vh-8rem)] overflow-y-auto custom-scrollbar">
            <ControlGroup title="Removal Mode">
              <div className="grid grid-cols-3 gap-2 p-1 bg-black rounded-xl border border-zinc-800">
                {(['chroma', 'ai', 'mask'] as const).map((mode) => (
                  <button 
                    key={mode}
                    onClick={() => setSettings(prev => ({ ...prev, mode }))}
                    className={cn(
                      "py-2 px-1 rounded-lg text-[10px] font-bold transition-all capitalize",
                      settings.mode === mode ? "bg-zinc-800 text-white shadow-lg" : "text-zinc-500 hover:text-zinc-300"
                    )}
                  >
                    {mode === 'chroma' ? 'Chroma' : mode === 'ai' ? 'AI' : 'Mask'}
                  </button>
                ))}
              </div>
            </ControlGroup>

            {settings.mode === 'mask' && (
              <ControlGroup title="Masking Tools" icon={Pencil}>
                <div className="grid grid-cols-4 gap-2 mb-4">
                  <button 
                    onClick={() => setSettings(prev => ({ ...prev, maskTool: 'brush' }))}
                    className={cn(
                      "p-3 rounded-xl border transition-all flex flex-col items-center gap-1",
                      settings.maskTool === 'brush' ? "border-emerald-500 bg-emerald-500/10 text-emerald-500" : "border-zinc-800 text-zinc-500 hover:border-zinc-700"
                    )}
                  >
                    <Pencil className="w-4 h-4" />
                    <span className="text-[8px] font-bold">Brush</span>
                  </button>
                  <button 
                    onClick={() => setSettings(prev => ({ ...prev, maskTool: 'polygon' }))}
                    className={cn(
                      "p-3 rounded-xl border transition-all flex flex-col items-center gap-1",
                      settings.maskTool === 'polygon' ? "border-emerald-500 bg-emerald-500/10 text-emerald-500" : "border-zinc-800 text-zinc-500 hover:border-zinc-700"
                    )}
                  >
                    <Hexagon className="w-4 h-4" />
                    <span className="text-[8px] font-bold">Poly</span>
                  </button>
                  <button 
                    onClick={() => setSettings(prev => ({ ...prev, maskTool: 'eraser' }))}
                    className={cn(
                      "p-3 rounded-xl border transition-all flex flex-col items-center gap-1",
                      settings.maskTool === 'eraser' ? "border-emerald-500 bg-emerald-500/10 text-emerald-500" : "border-zinc-800 text-zinc-500 hover:border-zinc-700"
                    )}
                  >
                    <Eraser className="w-4 h-4" />
                    <span className="text-[8px] font-bold">Eraser</span>
                  </button>
                  <button 
                    onClick={clearMask}
                    className="p-3 rounded-xl border border-zinc-800 text-zinc-500 hover:border-red-500/50 hover:text-red-400 transition-all flex flex-col items-center gap-1"
                  >
                    <Trash2 className="w-4 h-4" />
                    <span className="text-[8px] font-bold">Clear</span>
                  </button>
                </div>

                <button 
                  onClick={refineMaskWithAI}
                  disabled={isRefining}
                  className={cn(
                    "w-full py-3 mb-4 rounded-xl border transition-all flex items-center justify-center gap-2 text-xs font-bold",
                    isRefining 
                      ? "bg-zinc-800 border-zinc-700 text-zinc-500 cursor-not-allowed" 
                      : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20 hover:border-emerald-500"
                  )}
                >
                  {isRefining ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      AI Detecting Object...
                    </>
                  ) : (
                    <>
                      <Wand2 className="w-4 h-4" />
                      Magic AI Refine
                    </>
                  )}
                </button>

                {settings.maskTool === 'polygon' && polygonPoints.length > 0 && (
                  <div className="flex gap-2 mb-4">
                    <button 
                      onClick={finishPolygon}
                      className="flex-1 py-2 bg-emerald-500 text-black font-bold rounded-lg text-[10px]"
                    >
                      Finish Polygon
                    </button>
                    <button 
                      onClick={() => setPolygonPoints([])}
                      className="px-4 py-2 bg-zinc-800 text-zinc-400 font-bold rounded-lg text-[10px]"
                    >
                      Cancel
                    </button>
                  </div>
                )}

                <Slider 
                  label="Brush Size" 
                  value={settings.brushSize} 
                  min={1} max={200} step={1} 
                  onChange={(v: number) => setSettings(prev => ({ ...prev, brushSize: v }))} 
                />
                <Slider 
                  label="Feathering" 
                  value={settings.maskFeather} 
                  min={0} max={50} step={1} 
                  onChange={(v: number) => setSettings(prev => ({ ...prev, maskFeather: v }))} 
                />

                <div className="flex items-center justify-between p-3 bg-black rounded-xl border border-zinc-800">
                  <span className="text-[10px] font-bold text-zinc-400">Invert Mask</span>
                  <button 
                    onClick={() => setSettings(prev => ({ ...prev, invertMask: !prev.invertMask }))}
                    className={cn(
                      "w-10 h-5 rounded-full transition-all relative",
                      settings.invertMask ? "bg-emerald-500" : "bg-zinc-800"
                    )}
                  >
                    <div className={cn(
                      "absolute top-1 w-3 h-3 rounded-full bg-white transition-all",
                      settings.invertMask ? "left-6" : "left-1"
                    )} />
                  </button>
                </div>
              </ControlGroup>
            )}

            {settings.mode === 'chroma' && (
              <ControlGroup title="Chroma Settings">
                <div className="flex items-center gap-4 mb-4">
                  <div 
                    className="w-12 h-12 rounded-xl border-2 border-zinc-700 shadow-inner"
                    style={{ backgroundColor: `rgb(${settings.chromaColor.r}, ${settings.chromaColor.g}, ${settings.chromaColor.b})` }}
                  />
                  <div className="flex-1">
                    <p className="text-[10px] text-zinc-500 mb-1">Target Color</p>
                    <p className="text-xs font-mono">
                      RGB({settings.chromaColor.r}, {settings.chromaColor.g}, {settings.chromaColor.b})
                    </p>
                  </div>
                </div>
                <Slider 
                  label="Similarity" 
                  value={settings.similarity} 
                  min={0} max={1} step={0.01} 
                  onChange={(v: number) => setSettings(prev => ({ ...prev, similarity: v }))} 
                />
                <Slider 
                  label="Smoothness" 
                  value={settings.smoothness} 
                  min={0} max={1} step={0.01} 
                  onChange={(v: number) => setSettings(prev => ({ ...prev, smoothness: v }))} 
                />
                <Slider 
                  label="Spill Reduction" 
                  value={settings.spill} 
                  min={0} max={1} step={0.01} 
                  onChange={(v: number) => setSettings(prev => ({ ...prev, spill: v }))} 
                />
              </ControlGroup>
            )}

            <ControlGroup title="Background">
              <div className="flex flex-wrap gap-2">
                {(['transparent', 'color', 'image'] as const).map((type) => (
                  <button
                    key={type}
                    onClick={() => setSettings(prev => ({ ...prev, backgroundType: type }))}
                    className={cn(
                      "px-3 py-1.5 rounded-full text-[10px] font-bold border transition-all",
                      settings.backgroundType === type 
                        ? "bg-emerald-500/10 border-emerald-500 text-emerald-500" 
                        : "border-zinc-800 text-zinc-500 hover:border-zinc-700"
                    )}
                  >
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </button>
                ))}
              </div>

              {settings.backgroundType === 'color' && (
                <div className="flex items-center gap-3 mt-4 p-3 bg-black rounded-xl border border-zinc-800">
                  <input 
                    type="color" 
                    value={settings.backgroundColor}
                    onChange={(e) => setSettings(prev => ({ ...prev, backgroundColor: e.target.value }))}
                    className="w-8 h-8 bg-transparent border-none cursor-pointer"
                  />
                  <span className="text-xs font-mono text-zinc-400 uppercase">{settings.backgroundColor}</span>
                </div>
              )}

              {settings.backgroundType === 'image' && (
                <div className="mt-4 space-y-3">
                  {settings.backgroundImage ? (
                    <div className="relative aspect-video rounded-xl overflow-hidden border border-zinc-800">
                      <img src={settings.backgroundImage} className="w-full h-full object-cover" alt="bg" />
                      <button 
                        onClick={() => setSettings(prev => ({ ...prev, backgroundImage: null }))}
                        className="absolute top-2 right-2 p-1.5 bg-black/60 hover:bg-red-500 rounded-lg transition-colors"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  ) : (
                    <label className="flex flex-col items-center justify-center aspect-video bg-black border border-dashed border-zinc-800 rounded-xl cursor-pointer hover:border-emerald-500/50 transition-colors group">
                      <ImageIcon className="w-6 h-6 text-zinc-600 group-hover:text-emerald-500 transition-colors mb-2" />
                      <span className="text-[10px] text-zinc-500">Upload Background</span>
                      <input type="file" accept="image/*" className="hidden" onChange={handleBgImageUpload} />
                    </label>
                  )}
                </div>
              )}
            </ControlGroup>

            <button 
              onClick={() => {
                setVideoFile(null);
                setVideoUrl(null);
                setSettings(DEFAULT_SETTINGS);
                clearMask();
              }}
              className="w-full py-3 mt-4 border border-zinc-800 hover:bg-red-500/10 hover:border-red-500/20 hover:text-red-400 rounded-2xl text-xs font-bold transition-all flex items-center justify-center gap-2"
            >
              <Trash2 className="w-4 h-4" />
              Reset Project
            </button>
          </div>
        </div>
      </main>

    </div>
  );
}
