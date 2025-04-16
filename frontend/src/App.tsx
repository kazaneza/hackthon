import React, { useState, useRef } from 'react';
import { Mic, AlertCircle, StopCircle } from 'lucide-react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [transcription, setTranscription] = useState<string>('');
  const [gptResponse, setGptResponse] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
        setFile(audioFile);
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError('');
    } catch (err) {
      setError('Failed to access microphone. Please ensure microphone permissions are granted.');
      console.error(err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!file) {
      setError('Please make a recording first');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/transcribe', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setTranscription(response.data.transcription);
      setGptResponse(response.data.response);
    } catch (err) {
      setError('Failed to process recording. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 sm:text-4xl">
            Voice to Text with AI Response
          </h1>
          <p className="mt-3 text-xl text-gray-500 sm:mt-4">
            Record your voice and get instant transcription with AI-powered responses
          </p>
        </div>

        <div className="mt-12">
          <form onSubmit={handleSubmit} className="space-y-8">
            <div className="space-y-6">
              <div className="bg-white p-8 rounded-lg shadow">
                <div className="flex flex-col items-center justify-center">
                  <div className="mb-6 text-center">
                    <div className="flex items-center justify-center h-24 w-24 rounded-full bg-gray-100 mb-4">
                      {isRecording ? (
                        <div className="relative">
                          <div className="absolute inset-0 rounded-full bg-red-100 animate-ping"></div>
                          <StopCircle className="w-12 h-12 text-red-600 relative z-10" />
                        </div>
                      ) : (
                        <Mic className="w-12 h-12 text-gray-400" />
                      )}
                    </div>
                    <p className="text-sm text-gray-500">
                      {file ? 'Recording saved!' : 'Click the button below to start recording'}
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={isRecording ? stopRecording : startRecording}
                    className={`flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-full shadow-sm text-white ${
                      isRecording
                        ? 'bg-red-600 hover:bg-red-700'
                        : 'bg-green-600 hover:bg-green-700'
                    }`}
                  >
                    {isRecording ? (
                      <>
                        <StopCircle className="w-5 h-5 mr-2" />
                        Stop Recording
                      </>
                    ) : (
                      <>
                        <Mic className="w-5 h-5 mr-2" />
                        Start Recording
                      </>
                    )}
                  </button>
                </div>

                {error && (
                  <div className="mt-4 flex items-center justify-center text-red-600">
                    <AlertCircle className="w-5 h-5 mr-2" />
                    <span>{error}</span>
                  </div>
                )}

                <div className="mt-6">
                  <button
                    type="submit"
                    disabled={!file || loading || isRecording}
                    className={`w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white ${
                      !file || loading || isRecording
                        ? 'bg-gray-400 cursor-not-allowed'
                        : 'bg-blue-600 hover:bg-blue-700'
                    }`}
                  >
                    {loading ? 'Processing...' : 'Process Recording'}
                  </button>
                </div>
              </div>

              {(transcription || gptResponse) && (
                <div className="bg-white p-8 rounded-lg shadow space-y-6">
                  {transcription && (
                    <div>
                      <h2 className="text-lg font-medium text-gray-900 mb-4">
                        Transcription
                      </h2>
                      <div className="bg-gray-50 p-4 rounded-md">
                        <p className="text-gray-700 whitespace-pre-wrap">
                          {transcription}
                        </p>
                      </div>
                    </div>
                  )}
                  
                  {gptResponse && (
                    <div>
                      <h2 className="text-lg font-medium text-gray-900 mb-4">
                        AI Response
                      </h2>
                      <div className="bg-blue-50 p-4 rounded-md">
                        <p className="text-gray-700 whitespace-pre-wrap">
                          {gptResponse}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;