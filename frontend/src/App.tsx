import React, { useState, useRef, useCallback } from 'react';
import { Mic, AlertCircle, Volume2, Building2, Sparkles, MessageSquare, Bot } from 'lucide-react';
import axios from 'axios';

function App() {
  const [transcription, setTranscription] = useState<string>('');
  const [gptResponse, setGptResponse] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentSubtitle, setCurrentSubtitle] = useState<string>('');
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm'
      });
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError('');
    } catch (err) {
      setError('Failed to access microphone. Please ensure microphone permissions are granted.');
      console.error(err);
    }
  }, []);

  const stopRecording = useCallback(async () => {
    if (!mediaRecorderRef.current || !isRecording) return;

    return new Promise<void>((resolve) => {
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.onstop = async () => {
          const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
          const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
          
          if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
          }

          try {
            setLoading(true);
            const formData = new FormData();
            formData.append('file', audioFile);

            const response = await axios.post('http://localhost:8000/transcribe', formData, {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            });

            setTranscription(response.data.transcription);
            setGptResponse(response.data.response);
            playResponse(response.data.response);
          } catch (err) {
            setError('Failed to process recording. Please try again.');
            console.error(err);
          } finally {
            setLoading(false);
            resolve();
          }
        };

        mediaRecorderRef.current.stop();
        setIsRecording(false);
      }
    });
  }, [isRecording]);

  const playResponse = async (text: string) => {
    if (!text) return;
    
    try {
      setIsPlaying(true);
      setError('');
      
      const response = await axios({
        method: 'get',
        url: `http://localhost:8000/speak/${encodeURIComponent(text)}`,
        responseType: 'blob'
      });
      
      const audioBlob = new Blob([response.data], { type: 'audio/mpeg' });
      const audioUrl = URL.createObjectURL(audioBlob);
      
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        
        const words = text.split(' ');
        let wordIndex = 0;
        
        audioRef.current.ontimeupdate = () => {
          if (audioRef.current) {
            const wordsPerSecond = words.length / audioRef.current.duration;
            const currentWordIndex = Math.floor(audioRef.current.currentTime * wordsPerSecond);
            
            if (currentWordIndex !== wordIndex && currentWordIndex < words.length) {
              wordIndex = currentWordIndex;
              setCurrentSubtitle(words.slice(0, currentWordIndex + 1).join(' '));
            }
          }
        };
        
        audioRef.current.onended = () => {
          setIsPlaying(false);
          setCurrentSubtitle('');
          URL.revokeObjectURL(audioUrl);
        };
        
        try {
          await audioRef.current.play();
        } catch (playError) {
          setError('Failed to play audio. Please try again.');
          setIsPlaying(false);
          URL.revokeObjectURL(audioUrl);
        }
      }
    } catch (err) {
      setError('Failed to generate speech. Please try again.');
      setIsPlaying(false);
      console.error(err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-radial from-bk-light via-bk-gray to-bk-gray/50">
      {/* Header */}
      <div className="bg-gradient-to-r from-bk-blue to-bk-accent">
        <div className="max-w-6xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="bg-white p-2 rounded-lg shadow-lg">
                <Building2 className="w-8 h-8 text-bk-blue" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">
                  Bank of Kigali
                </h1>
                <p className="text-blue-100 text-sm">Meet Alice, Your AI Assistant</p>
              </div>
            </div>
            <div className="hidden sm:flex items-center space-x-2 bg-white/10 px-4 py-2 rounded-full text-white text-sm">
              <Bot className="w-4 h-4" />
              <span>Alice AI</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-bk-dark mb-4">
            Hi, I'm Alice 👋
          </h2>
          <p className="text-lg text-bk-dark/60 max-w-2xl mx-auto">
            I'm your personal banking assistant. Ask me about accounts, transfers, loans, or any banking service. Press and hold the microphone to speak with me.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-[1fr,2fr]">
          <div className="bg-white rounded-2xl shadow-xl border border-bk-blue/5 p-8 flex flex-col items-center justify-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-radial from-transparent to-bk-gray/5"></div>
            <div className="relative z-10">
              <div className="mb-8 text-center">
                <button
                  onMouseDown={startRecording}
                  onMouseUp={stopRecording}
                  onMouseLeave={stopRecording}
                  onTouchStart={startRecording}
                  onTouchEnd={stopRecording}
                  disabled={loading}
                  className={`group relative flex items-center justify-center h-32 w-32 rounded-full transition-all duration-500 ${
                    isRecording
                      ? 'bg-red-500 scale-110 shadow-lg shadow-red-500/20'
                      : loading
                      ? 'bg-bk-gray cursor-not-allowed'
                      : 'bg-bk-blue hover:bg-bk-accent hover:scale-105 shadow-lg shadow-bk-blue/20'
                  }`}
                >
                  <div className={`absolute inset-0 rounded-full ${isRecording ? 'animate-pulse-slow bg-red-400/20' : 'group-hover:animate-pulse-slow bg-bk-accent/20'}`}></div>
                  <Mic className={`w-12 h-12 ${
                    isRecording
                      ? 'text-white'
                      : loading
                      ? 'text-gray-400'
                      : 'text-white'
                  }`} />
                </button>
                <p className="mt-6 text-sm font-medium text-bk-dark/60">
                  {loading ? 'Processing...' : isRecording ? 'I\'m listening...' : 'Hold to talk with Alice'}
                </p>
              </div>

              {error && (
                <div className="mt-4 flex items-center justify-center text-red-600 bg-red-50 p-4 rounded-xl">
                  <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
                  <span className="text-sm">{error}</span>
                </div>
              )}
            </div>
          </div>

          <div className="space-y-6">
            {isPlaying && (
              <div className="bg-white rounded-2xl shadow-xl border border-bk-blue/5 p-8">
                <div className="flex items-center space-x-3 mb-6">
                  <div className="bg-blue-100 p-2 rounded-lg">
                    <Volume2 className="w-5 h-5 text-bk-blue animate-pulse" />
                  </div>
                  <h2 className="text-xl font-semibold text-bk-dark">Alice is speaking</h2>
                </div>
                <div className="bg-gradient-to-r from-bk-gray to-bk-gray/50 p-6 rounded-xl">
                  <p className="text-bk-dark text-lg leading-relaxed">
                    {currentSubtitle}
                  </p>
                </div>
              </div>
            )}

            {(transcription || gptResponse) && !isPlaying && (
              <div className="space-y-6">
                {transcription && (
                  <div className="bg-white rounded-2xl shadow-xl border border-bk-blue/5 p-8">
                    <div className="flex items-center space-x-3 mb-6">
                      <div className="bg-blue-100 p-2 rounded-lg">
                        <MessageSquare className="w-5 h-5 text-bk-blue" />
                      </div>
                      <h2 className="text-xl font-semibold text-bk-dark">You Said</h2>
                    </div>
                    <div className="bg-gradient-to-r from-bk-gray to-bk-gray/50 p-6 rounded-xl">
                      <p className="text-bk-dark text-lg leading-relaxed">{transcription}</p>
                    </div>
                  </div>
                )}
                
                {gptResponse && (
                  <div className="bg-white rounded-2xl shadow-xl border border-bk-blue/5 p-8">
                    <div className="flex items-center space-x-3 mb-6">
                      <div className="bg-blue-100 p-2 rounded-lg">
                        <Bot className="w-5 h-5 text-bk-blue" />
                      </div>
                      <h2 className="text-xl font-semibold text-bk-dark">Alice's Response</h2>
                    </div>
                    <div className="bg-gradient-to-r from-bk-gray to-bk-gray/50 p-6 rounded-xl">
                      <p className="text-bk-dark text-lg leading-relaxed">{gptResponse}</p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      <audio ref={audioRef} className="hidden" />
    </div>
  );
}

export default App;