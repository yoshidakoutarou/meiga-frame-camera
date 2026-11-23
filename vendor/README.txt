同梱しているライブラリとモデル（すべて端末の中だけで使い、外部へは何も送りません）

mediabunny.mjs
  Mediabunny 1.66.2（https://mediabunny.dev/）を、使う部分だけ esbuild でまとめたもの
  ライセンス：MPL-2.0（MEDIABUNNY-LICENSE.txt）
  取り込んだもの：Input, BlobSource, MP4, QTFF, WEBM, VideoSampleSink, Output, Mp4OutputFormat,
                  BufferTarget, CanvasSource, QUALITY_HIGH, canEncodeVideo

mediapipe/
  @mediapipe/tasks-vision 1.2.3（https://github.com/google-ai-edge/mediapipe）
  ライセンス：Apache-2.0

models/selfie_multiclass_256x256.tflite
  MediaPipe Image Segmenter のマルチクラスモデル（背景・髪・体の肌・顔の肌・服・その他）
  入手元：https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/
  ライセンス：Apache-2.0
