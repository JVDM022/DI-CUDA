const {app,BrowserWindow,dialog,shell}=require('electron');
const {spawn}=require('node:child_process');
const path=require('node:path');
const fs=require('node:fs');
const http=require('node:http');
let window,backend,port,token,closing=false;
const lock=app.requestSingleInstanceLock();
if(!lock){app.quit();}else{
 app.on('second-instance',()=>{if(window){if(window.isMinimized())window.restore();window.show();window.focus();}});
 app.whenReady().then(start).catch(fail);
}
function fail(error){dialog.showErrorBox('Kamiak DI could not start',String(error));if(backend)backend.kill();app.exit(1);}
function request(route,method='GET'){
 return new Promise((resolve,reject)=>{const body=method==='POST'?'{}':null;const req=http.request({host:'127.0.0.1',port,path:'/api/'+route,method,headers:{Host:'127.0.0.1:'+port,'X-Kamiak-Token':token,...(body?{'Content-Type':'application/json','Content-Length':2}:{})}},res=>{let raw='';res.on('data',b=>raw+=b);res.on('end',()=>{try{const data=JSON.parse(raw);res.statusCode<300?resolve(data):reject(Error(data.error||'Local request failed'));}catch(e){reject(e);}});});req.on('error',reject);req.setTimeout(5000,()=>req.destroy(Error('Local service did not respond')));if(body)req.write(body);req.end();});
}
async function start(){
 const base=app.isPackaged?path.join(process.resourcesPath,'app','payload'):path.resolve(__dirname,'..','..');
 const bundled=path.join(base,'python',process.platform==='win32'?'python.exe':'bin/python3');
 const candidates=[bundled,'/Library/Frameworks/Python.framework/Versions/Current/bin/python3','/usr/local/bin/python3','/opt/homebrew/bin/python3'];
 const python=candidates.find(p=>fs.existsSync(p));
 if(!python)throw Error('Python runtime is missing. Use the packaged Kamiak DI application.');
 const script=path.join(base,'desktop','kamiak_web.py');
 window=new BrowserWindow({width:1380,height:940,minWidth:980,minHeight:700,title:'Kamiak DI',backgroundColor:'#f5f7fa',show:false,autoHideMenuBar:true,webPreferences:{nodeIntegration:false,contextIsolation:true,sandbox:true}});
 await window.loadURL('data:text/html;charset=utf-8,'+encodeURIComponent('<html><body style="font:16px -apple-system,sans-serif;background:#f5f7fa;color:#435778;display:grid;place-items:center;height:90vh"><div><h1>Kamiak DI</h1><p>Opening your research workspace…</p></div></body></html>'));
 window.show();
 let ready=false,buffer='',errors='';
 backend=spawn(python,['-I',script,'--desktop'],{cwd:base,env:{...process.env,PYTHONUNBUFFERED:'1',PATH:path.dirname(python)+path.delimiter+(process.env.PATH||'/usr/bin:/bin:/usr/sbin:/sbin')},stdio:['ignore','pipe','pipe'],windowsHide:true});
 const timeout=setTimeout(()=>{if(!ready)fail(Error('The local service did not start. '+errors.slice(-2000)));},30000);
 backend.on('error',fail);
 backend.stderr.on('data',data=>errors=(errors+data.toString()).slice(-10000));
 backend.stdout.on('data',async data=>{buffer+=data.toString();let end;while((end=buffer.indexOf('\n'))>=0){const line=buffer.slice(0,end);buffer=buffer.slice(end+1);if(!ready&&line.startsWith('{')){try{const info=JSON.parse(line);if(info.kamiak_ready&&Number.isInteger(info.port)&&typeof info.token==='string'){ready=true;clearTimeout(timeout);port=info.port;token=info.token;const origin='http://127.0.0.1:'+port;
 window.webContents.setWindowOpenHandler(({url})=>{if(url.startsWith('https://github.com/JVDM022/DI-CUDA/'))shell.openExternal(url);return {action:'deny'};});
 window.webContents.on('will-navigate',(event,url)=>{if(new URL(url).origin!==origin)event.preventDefault();});
 await window.loadURL(origin+'/#'+token);window.focus();}}catch(e){fail(e);}}}});
 backend.on('exit',code=>{clearTimeout(timeout);if(!closing){if(ready){closing=true;app.quit();}else fail(Error('Local service exited: '+code+'\n'+errors.slice(-3000)));}});
 window.on('close',event=>{if(closing)return;event.preventDefault();if(!ready){closing=true;if(backend)backend.kill();app.quit();return;}
 request('state').then(state=>{if(state.busy){dialog.showMessageBox(window,{type:'info',message:'An action is still in progress.',detail:'Wait for the upload, download or file selection to finish before closing.'});return;}closing=true;request('shutdown','POST').catch(()=>{}).finally(()=>app.quit());}).catch(()=>{closing=true;app.quit();});});
 app.on('activate',()=>{if(window){window.show();window.focus();}});
}
app.on('will-quit',()=>{if(backend)backend.kill();});
app.on('window-all-closed',()=>{if(closing)app.quit();});
