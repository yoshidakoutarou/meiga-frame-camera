// 仕上げ処理（録画後）：穴の中の「服」の色だけを、絵から拾った色に寄せる
// 動画の読み込み・書き出し：Mediabunny（WebCodecs）／服の判定：MediaPipe Image Segmenter（マルチクラス）
// 色の変換：WebGL2で、Lab色空間の L・a・b を「拾った色 − 服の平均色」だけずらす（陰影は残る）
import {Input,BlobSource,MP4,QTFF,WEBM,VideoSampleSink,Output,Mp4OutputFormat,BufferTarget,CanvasSource,QUALITY_HIGH,canEncodeVideo} from './vendor/mediabunny.mjs';

const SEG_MAX=384;        // 判定に使う画像の長辺（モデル自体は256×256で判定する）
const MASK_EMA=.5;        // コマ間で判定をなめらかにする割合（前のコマの重み）
const MEAN_EMA=.9;        // 服の平均色のなめらかさ（前のコマの重み）
const url=p=>new URL(p,import.meta.url).href;

// ---- 色空間（sRGB ⇔ Lab, D65）。平均色の計算に使う ----
const lin=c=>(c/=255)<=.04045?c/12.92:Math.pow((c+.055)/1.055,2.4);
const fLab=t=>t>216/24389?Math.cbrt(t):(24389/27*t+16)/116;
export function rgb2lab(r,g,b){
  r=lin(r);g=lin(g);b=lin(b);
  const x=(.4124*r+.3576*g+.1805*b)/.95047,y=.2126*r+.7152*g+.0722*b,z=(.0193*r+.1192*g+.9505*b)/1.08883;
  const fx=fLab(x),fy=fLab(y),fz=fLab(z);return [116*fy-16,500*(fx-fy),200*(fy-fz)];
}

// ---- MediaPipe（初回だけ読み込む）----
let segP=null,mpTs=0;
export function loadSegmenter(){
  if(!segP)segP=(async()=>{
    const {FilesetResolver,ImageSegmenter}=await import('./vendor/mediapipe/vision_bundle.mjs');
    const fs=await FilesetResolver.forVisionTasks(url('./vendor/mediapipe/wasm'));
    const opt=d=>({baseOptions:{modelAssetPath:url('./vendor/models/selfie_multiclass_256x256.tflite'),delegate:d},
      runningMode:'VIDEO',outputCategoryMask:false,outputConfidenceMasks:true});
    let seg;
    try{seg=await ImageSegmenter.createFromOptions(fs,opt('GPU'));}
    catch(e){seg=await ImageSegmenter.createFromOptions(fs,opt('CPU'));}
    return {seg,clothes:seg.getLabels().indexOf('clothes')};
  })().catch(e=>{segP=null;throw e;});
  return segP;
}

// ---- WebGL2：コマ全体を描き、穴の中だけ服の色を寄せる ----
const VS=`#version 300 es
in vec2 p;out vec2 vUv;
void main(){vUv=vec2((p.x+1.)*.5,(1.-p.y)*.5);gl_Position=vec4(p,0.,1.);}`;
const FS=`#version 300 es
precision highp float;
uniform sampler2D uFrame,uMask;uniform vec4 uHole;uniform vec3 uDelta;uniform float uK;
in vec2 vUv;out vec4 o;
vec3 lin(vec3 c){return mix(c/12.92,pow((c+.055)/1.055,vec3(2.4)),step(.04045,c));}
vec3 gam(vec3 c){c=max(c,0.);return mix(12.92*c,1.055*pow(c,vec3(1./2.4))-.055,step(.0031308,c));}
float f(float t){return t>216./24389.?pow(t,1./3.):(24389./27.*t+16.)/116.;}
float fi(float t){return t*t*t>216./24389.?t*t*t:(116.*t-16.)/(24389./27.);}
vec3 toLab(vec3 c){c=lin(c);
  float x=dot(vec3(.4124,.3576,.1805),c)/.95047,y=dot(vec3(.2126,.7152,.0722),c),z=dot(vec3(.0193,.1192,.9505),c)/1.08883;
  float fx=f(x),fy=f(y),fz=f(z);return vec3(116.*fy-16.,500.*(fx-fy),200.*(fy-fz));}
vec3 toRgb(vec3 l){float fy=(l.x+16.)/116.,fx=fy+l.y/500.,fz=fy-l.z/200.;
  vec3 q=vec3(fi(fx)*.95047,fi(fy),fi(fz)*1.08883);
  vec3 r=vec3(dot(vec3(3.2406,-1.5372,-.4986),q),dot(vec3(-.9689,1.8758,.0415),q),dot(vec3(.0557,-.204,1.057),q));
  return clamp(gam(r),0.,1.);}
void main(){
  vec4 c=texture(uFrame,vUv);
  vec2 m=(vUv-uHole.xy)/uHole.zw;
  float k=0.;
  if(m.x>=0.&&m.y>=0.&&m.x<=1.&&m.y<=1.)k=texture(uMask,m).r*uK;
  if(k>.002)c.rgb=toRgb(toLab(c.rgb)+k*uDelta);
  o=vec4(c.rgb,1.);
}`;
function makeGL(w,h){
  const cv=document.createElement('canvas');cv.width=w;cv.height=h;
  const gl=cv.getContext('webgl2',{preserveDrawingBuffer:true,alpha:false,antialias:false});
  if(!gl)throw new Error('このブラウザはWebGL2に対応していないため、仕上げ処理を使えません。');
  const sh=(t,s)=>{const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(o));return o;};
  const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,VS));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,FS));gl.linkProgram(pr);
  if(!gl.getProgramParameter(pr,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(pr));
  gl.useProgram(pr);
  const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  const loc=gl.getAttribLocation(pr,'p');gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
  const tex=unit=>{const t=gl.createTexture();gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,t);
    for(const [k,v] of [[gl.TEXTURE_MIN_FILTER,gl.LINEAR],[gl.TEXTURE_MAG_FILTER,gl.LINEAR],[gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE]])gl.texParameteri(gl.TEXTURE_2D,k,v);return t;};
  const tFrame=tex(0),tMask=tex(1);
  gl.uniform1i(gl.getUniformLocation(pr,'uFrame'),0);gl.uniform1i(gl.getUniformLocation(pr,'uMask'),1);
  const uHole=gl.getUniformLocation(pr,'uHole'),uDelta=gl.getUniformLocation(pr,'uDelta'),uK=gl.getUniformLocation(pr,'uK');
  gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);gl.viewport(0,0,w,h);
  return {cv,render(frame,mask,mw,mh,hole,delta,k){
    gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,tFrame);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,frame);
    gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,tMask);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.R8,mw,mh,0,gl.RED,gl.UNSIGNED_BYTE,mask);
    gl.uniform4f(uHole,hole.x,hole.y,hole.w,hole.h);gl.uniform3f(uDelta,delta[0],delta[1],delta[2]);gl.uniform1f(uK,k);
    gl.drawArrays(gl.TRIANGLES,0,3);
  }};
}

/**
 * 動画を仕上げる
 * @param {Blob} blob 録画した動画
 * @param {{x,y,w,h}} hole 穴の位置（画面に対する比率 0〜1）
 * @param {{rgb:number[],k:number}} cloth 袖の色（RGB 0〜255）と寄せる強さ（0〜1）
 * @param {(p:number,msg:string)=>void} onProgress 進み具合（0〜1）
 * @param {()=>boolean} canceled 中止されたら true を返す関数
 * @returns {Promise<Blob|null>} 仕上げた動画（中止したら null）
 */
export async function finishVideo({blob,hole,cloth,onProgress,canceled}){
  onProgress(0,'準備中（初回は約30MBを読み込みます）');
  const {seg,clothes}=await loadSegmenter();
  const input=new Input({source:new BlobSource(blob),formats:[MP4,QTFF,WEBM]});
  const track=await input.getPrimaryVideoTrack();
  if(!track)throw new Error('動画の映像が見つかりませんでした。');
  const W=track.displayWidth,H=track.displayHeight;
  if(!(await canEncodeVideo('avc',{width:W,height:H})))throw new Error('この端末ではMP4（H.264）に書き出せません。');
  const total=await track.computeDuration();

  // 作業用のキャンバス：コマ全体（2D）、判定用の縮小（2D）、出力（WebGL2）
  const frame=document.createElement('canvas');frame.width=W;frame.height=H;
  const fctx=frame.getContext('2d');
  const hx=hole.x*W,hy=hole.y*H,hw=hole.w*W,hh=hole.h*H;
  const s=Math.min(1,SEG_MAX/Math.max(hw,hh));
  const sw=Math.max(16,Math.round(hw*s)),sh=Math.max(16,Math.round(hh*s));
  const small=document.createElement('canvas');small.width=sw;small.height=sh;
  const sctx=small.getContext('2d',{willReadFrequently:true});
  const gl=makeGL(W,H);

  const output=new Output({format:new Mp4OutputFormat({fastStart:'in-memory'}),target:new BufferTarget()});
  const source=new CanvasSource(gl.cv,{codec:'avc',quality:QUALITY_HIGH});
  output.addVideoTrack(source);
  await output.start();

  const target=cloth.lab||rgb2lab(...cloth.rgb);
  const sink=new VideoSampleSink(track);
  let ema=null,mean=null,t0=null,n=0;
  const mask8=new Uint8Array(sw*sh);
  // 工程ごとの時間（ms）。実機での速さの確認用
  const st=finishVideo.stats={draw:0,seg:0,mean:0,gl:0,encode:0,frames:0};
  let tt=performance.now();const lap=k=>{const t=performance.now();st[k]+=t-tt;tt=t;};
  try{
    for await(const sample of sink.samples()){
      try{
        if(canceled()){await output.cancel();return null;}
        if(t0===null)t0=sample.timestamp;
        tt=performance.now();
        sample.draw(fctx,0,0,W,H);
        sctx.drawImage(frame,hx,hy,hw,hh,0,0,sw,sh);
        lap('draw');
        // 服の判定（コマ間でなめらかに）
        mpTs+=33;
        const r=seg.segmentForVideo(small,mpTs);
        const m=r.confidenceMasks[clothes].getAsFloat32Array();
        if(!ema)ema=Float32Array.from(m);
        else for(let i=0;i<ema.length;i++)ema[i]=ema[i]*MASK_EMA+m[i]*(1-MASK_EMA);
        r.close();
        lap('seg');
        // 服の平均色（Lab）。服が十分に写っているコマだけで更新し、急に変わらないようにする
        const px=sctx.getImageData(0,0,sw,sh).data;
        let L=0,A=0,B=0,c=0;
        for(let y=0;y<sh;y+=2)for(let x=0;x<sw;x+=2){
          const i=y*sw+x;if(ema[i]<.6)continue;
          const l=rgb2lab(px[i*4],px[i*4+1],px[i*4+2]);L+=l[0];A+=l[1];B+=l[2];c++;
        }
        if(c>200){const cur=[L/c,A/c,B/c];mean=mean?mean.map((v,j)=>v*MEAN_EMA+cur[j]*(1-MEAN_EMA)):cur;}
        for(let i=0;i<mask8.length;i++)mask8[i]=Math.round(Math.min(1,Math.max(0,ema[i]))*255);
        const delta=mean?[target[0]-mean[0],target[1]-mean[1],target[2]-mean[2]]:[0,0,0];
        lap('mean');
        gl.render(frame,mask8,sw,sh,hole,delta,mean?cloth.k:0);
        lap('gl');
        await source.add(sample.timestamp-t0,sample.duration);
        lap('encode');
        n++;st.frames=n;
        onProgress(Math.min(1,(sample.timestamp-t0+sample.duration)/total),n+'コマ目を処理中');
      }finally{sample.close();}
    }
    onProgress(1,'書き出し中');
    await output.finalize();
  }catch(e){
    if(output.state!=='canceled'&&output.state!=='finalized')await output.cancel().catch(()=>{});
    throw e;
  }
  return new Blob([output.target.buffer],{type:'video/mp4'});
}
