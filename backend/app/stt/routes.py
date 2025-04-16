from fastapi import WebSocket, WebSocketDisconnect
from . import router
from .audio import AudioBuffer
from .transcriber import transcribe_audio
import numpy as np

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = AudioBuffer([])
    
    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                
                # Convert bytes to numpy array
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                # Add samples and check if we should process
                if audio_buffer.add_samples(audio_chunk):
                    audio_data = audio_buffer.get_audio_data()
                    
                    # Only transcribe if we have enough audio
                    if len(audio_data) > RATE * 0.5:  # At least 0.5 seconds
                        transcription = await transcribe_audio(audio_data)
                        if transcription:
                            await websocket.send_json({
                                "type": "transcription",
                                "text": transcription
                            })
                    
                # Send audio level for visualization
                rms = np.sqrt(np.mean(np.square(audio_chunk)))
                await websocket.send_json({
                    "type": "audio_level",
                    "level": float(rms)
                })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"Error processing audio: {e}")
                break
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket connection closed")