import React, { useState, useRef, useCallback } from 'react';
import { Mic, AlertCircle, Volume2 } from 'lucide-react';
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
          
          // Clean up the stream
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
            
            // Automatically play the response
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
        
        // Split response into words for subtitle animation
        const words = text.split(' ');
        let wordIndex = 0;
        
        audioRef.current.ontimeupdate = () => {
          if (audioRef.current) {
            // Roughly estimate word timing based on audio duration
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
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 sm:text-4xl">
            Voice Chat with AI
          </h1>
          <p className="mt-3 text-xl text-gray-500 sm:mt-4">
            Press and hold to speak with the AI assistant
          </p>
        </div>

        <div className="mt-12">
          <div className="space-y-6">
            <div className="bg-white p-8 rounded-lg shadow">
              <div className="flex flex-col items-center justify-center">
                <div className="mb-6 text-center">
                  <button
                    onMouseDown={startRecording}
                    onMouseUp={stopRecording}
                    onMouseLeave={stopRecording}
                    onTouchStart={startRecording}
                    onTouchEnd={stopRecording}
                    disabled={loading}
                    className={`flex items-center justify-center h-24 w-24 rounded-full transition-all duration-200 ${
                      isRecording
                        ? 'bg-red-100 scale-110'
                        : loading
                        ? 'bg-gray-100 cursor-not-allowed'
                        : 'bg-blue-100 hover:bg-blue-200'
                    }`}
                  >
                    <Mic className={`w-12 h-12 ${
                      isRecording
                        ? 'text-red-600'
                        : loading
                        ? 'text-gray-400'
                        : 'text-blue-600'
                    }`} />
                  </button>
                  <p className="mt-4 text-sm text-gray-500">
                    {loading ? 'Processing...' : isRecording ? 'Recording...' : 'Press and hold to speak'}
                  </p>
                </div>

                {error && (
                  <div className="mt-4 flex items-center justify-center text-red-600">
                    <AlertCircle className="w-5 h-5 mr-2" />
                    <span>{error}</span>
                  </div>
                )}
              </div>
            </div>

            {isPlaying && (
              <div className="bg-white p-8 rounded-lg shadow">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-medium text-gray-900">AI Speaking</h2>
                  <Volume2 className="w-5 h-5 text-blue-600 animate-pulse" />
                </div>
                <div className="bg-blue-50 p-4 rounded-md">
                  <p className="text-gray-700 whitespace-pre-wrap">
                    {currentSubtitle}
                  </p>
                </div>
              </div>
            )}

            {(transcription || gptResponse) && !isPlaying && (
              <div className="bg-white p-8 rounded-lg shadow space-y-6">
                {transcription && (
                  <div>
                    <h2 className="text-lg font-medium text-gray-900 mb-4">
                      You Said
                    </h2>
                    <div className="bg-gray-50 p-4 rounded-md">
                      <p className="text-gray-700">{transcription}</p>
                    </div>
                  </div>
                )}
                
                {gptResponse && (
                  <div>
                    <h2 className="text-lg font-medium text-gray-900 mb-4">
                      AI Response
                    </h2>
                    <div className="bg-blue-50 p-4 rounded-md">
                      <p className="text-gray-700">{gptResponse}</p>
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