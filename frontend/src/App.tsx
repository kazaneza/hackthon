import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Mic, AlertCircle, Volume2, Building2,
  Bot, MessageSquare, StopCircle,
} from 'lucide-react';
import { api, nlpApi, sessionManager } from './utils/sessionManager';

interface Transaction {
  date: string;
  description: string;
  amount: number;
  balance: number;
  reference?: string;
}

interface APIResponse {
  transcription?: string;
  response?: string;
  transactions?: Transaction[];
  session_id?: string;
}

function App() {
  const [transcription, setTranscription] = useState('');
  const [gptResponse, setGptResponse] = useState('');
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentSubtitle, setCurrentSubtitle] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Load session ID on component mount
  useEffect(() => {
    let currentSession = sessionManager.getSessionId();
    
    // If no session exists, create a permanent one
    if (!currentSession) {
      currentSession = `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      sessionManager.setSessionId(currentSession);
    }
    
    setSessionId(currentSession);
    console.log('Current session:', currentSession);
  }, []);

  const processWithNLP = async (text: string) => {
    try {
      // Ensure we have a session ID
      const currentSessionId = sessionManager.getSessionId();
      console.log('NLP request with session ID:', currentSessionId);
      
      const { data } = await nlpApi.post('/chat', {
        messages: [{ role: 'user', content: text }],
        service_category: 'general',
        user_id: currentSessionId // Explicitly pass the user_id
      });
      
      return data.response || data.message;
    } catch (err) {
      console.error('NLP processing error:', err);
      throw err;
    }
  };

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      recorder.start();
      setIsRecording(true);
      setError('');
    } catch (err) {
      setError('Microphone access denied. Please allow microphone.');
      console.error(err);
    }
  }, []);

  const stopRecording = useCallback(async () => {
    if (!mediaRecorderRef.current || !isRecording) return;
    return new Promise<void>((resolve) => {
      mediaRecorderRef.current!.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const file = new File([blob], 'recording.webm', { type: 'audio/webm' });
        streamRef.current?.getTracks().forEach((t) => t.stop());

        try {
          setLoading(true);
          const fd = new FormData();
          fd.append('file', file);
          
          // Ensure we have a session ID
          const currentSessionId = sessionManager.getSessionId();
          console.log('Transcription request with session ID:', currentSessionId);
          
          // First get transcription
          const { data: transcriptionData } = await api.post<APIResponse>('/transcribe', fd);
          setTranscription(transcriptionData.transcription || '');
          
          // Update local session state if it changed
          const updatedSession = sessionManager.getSessionId();
          if (updatedSession !== sessionId) {
            setSessionId(updatedSession);
          }
          
          // Then process with NLP - this will use the session ID
          if (transcriptionData.transcription) {
            const nlpResponse = await processWithNLP(transcriptionData.transcription);
            setGptResponse(nlpResponse);
            playResponse(nlpResponse);
          }

          // Set transactions if available
          if (transcriptionData.transactions) {
            setTransactions(transcriptionData.transactions);
          }
        } catch (err) {
          setError('Failed to process recording.');
          console.error(err);
        } finally {
          setLoading(false);
          resolve();
        }
      };
      mediaRecorderRef.current!.stop();
      setIsRecording(false);
    });
  }, [isRecording, sessionId]);

  const stopPlaying = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
      setCurrentSubtitle('');
    }
  };

  const playResponse = async (text: string) => {
    if (!text) return;
    try {
      setIsPlaying(true);
      setError('');
      const { data } = await api.get(`/speak/${encodeURIComponent(text)}`, { responseType: 'blob' });
      const url = URL.createObjectURL(new Blob([data], { type: 'audio/mpeg' }));
      
      if (!audioRef.current) return;

      const words = text.split(' ');
      let idx = 0;
      audioRef.current.src = url;
      audioRef.current.ontimeupdate = () => {
        const wps = words.length / audioRef.current!.duration;
        const cur = Math.floor(audioRef.current!.currentTime * wps);
        if (cur !== idx && cur < words.length) {
          idx = cur;
          setCurrentSubtitle(words.slice(0, cur + 1).join(' '));
        }
      };
      audioRef.current.onended = () => {
        setIsPlaying(false);
        setCurrentSubtitle('');
        URL.revokeObjectURL(url);
      };
      await audioRef.current.play();
    } catch (err) {
      setError('Failed to play audio.');
      setIsPlaying(false);
      console.error(err);
    }
  };

  // Optional: Add a button to clear session (for debugging)
  const clearSession = () => {
    sessionManager.clearSession();
    // Create a new session ID
    const newSessionId = `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    sessionManager.setSessionId(newSessionId);
    setSessionId(newSessionId);
    
    setTranscription('');
    setGptResponse('');
    setTransactions([]);
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
              {sessionId && (
                <span className="text-xs opacity-75">
                  Session: {sessionId.substring(0, 8)}...
                </span>
              )}
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
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center space-x-3">
                    <div className="bg-blue-100 p-2 rounded-lg">
                      <Volume2 className="w-5 h-5 text-bk-blue animate-pulse" />
                    </div>
                    <h2 className="text-xl font-semibold text-bk-dark">Alice is speaking</h2>
                  </div>
                  <button
                    onClick={stopPlaying}
                    className="bg-red-100 p-2 rounded-lg hover:bg-red-200 transition-colors"
                  >
                    <StopCircle className="w-5 h-5 text-red-600" />
                  </button>
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

                {transactions.length > 0 && (
                  <div className="bg-white rounded-2xl shadow-xl border border-bk-blue/5 p-8">
                    <div className="flex items-center space-x-3 mb-6">
                      <div className="bg-blue-100 p-2 rounded-lg">
                        <Bot className="w-5 h-5 text-bk-blue" />
                      </div>
                      <h2 className="text-xl font-semibold text-bk-dark">Recent Transactions</h2>
                    </div>
                    <div className="space-y-4">
                      {transactions.map((tx, index) => (
                        <div key={index} className="bg-gradient-to-r from-bk-gray to-bk-gray/50 p-4 rounded-xl">
                          <div className="flex justify-between items-center">
                            <div>
                              <p className="font-medium text-bk-dark">{tx.description}</p>
                              <p className="text-sm text-bk-dark/60">{tx.date}</p>
                            </div>
                            <div className="text-right">
                              <p className={`font-medium ${tx.amount < 0 ? 'text-red-600' : 'text-green-600'}`}>
                                {tx.amount < 0 ? '-' : '+'}RWF {Math.abs(tx.amount).toLocaleString()}
                              </p>
                              <p className="text-sm text-bk-dark/60">
                                Balance: RWF {tx.balance.toLocaleString()}
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      <audio ref={audioRef} className="hidden" />
      
      {/* Optional: Debug session button - remove in production */}
      {process.env.NODE_ENV === 'development' && (
        <button
          onClick={clearSession}
          className="fixed bottom-4 right-4 bg-red-500 text-white p-2 rounded hover:bg-red-600 transition-colors z-50"
          title="Clear Session (Development Only)"
        >
          Clear Session
        </button>
      )}
    </div>
  );
}

export default App;