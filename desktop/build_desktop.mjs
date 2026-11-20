/** Build the native Mac GUI from a checksum-pinned standalone Python archive. */
import {packager} from './shell/node_modules/@electron/packager/dist/index.js';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const desktop=path.dirname(fileURLToPath(import.meta.url));
const root=path.dirname(desktop);
const archive=process.argv[2];
if(!archive)throw Error('Usage: node desktop/build_desktop.mjs MAC_PYTHON_RUNTIME.tar.gz');
const sha=crypto.createHash('sha256').update(await fs.readFile(archive)).digest('hex');
if(sha!=='2f11b8aa3726bdbd87dd9eb76568412faddf4f47e26913fa45d6a2e424972acd')throw Error('Unexpected Python runtime checksum');
const stage=await fs.mkdtemp(path.join(os.tmpdir(),'kamiak-desktop-'));
try{
 for(const name of ['main.cjs','package.json'])await fs.copyFile(path.join(desktop,'shell',name),path.join(stage,name));
 const payload=path.join(stage,'payload');await fs.mkdir(path.join(payload,'desktop'),{recursive:true});
 for(const name of ['kamiak_di.py','kamiak_web.py'])await fs.copyFile(path.join(desktop,name),path.join(payload,'desktop',name));
 await fs.cp(path.join(desktop,'web-assets'),path.join(payload,'desktop','web-assets'),{recursive:true});
 await fs.mkdir(path.join(payload,'tools'),{recursive:true});await fs.copyFile(path.join(root,'tools/merge_cobalt.py'),path.join(payload,'tools/merge_cobalt.py'));
 await fs.cp(path.join(root,'SHTdatabase-master'),path.join(payload,'SHTdatabase-master'),{recursive:true,filter:src=>!src.includes('/.git/')&&!src.endsWith('/.git')});
 for(const name of ['LICENSE'])await fs.copyFile(path.join(root,name),path.join(payload,name));
 execFileSync('tar',['-xzf',path.resolve(archive),'-C',payload]);
 const electronVersion=JSON.parse(await fs.readFile(path.join(desktop,'shell/node_modules/electron/package.json'),'utf8')).version;
 const builds=await packager({dir:stage,out:path.join(root,'dist/native'),name:'Kamiak DI',platform:'darwin',arch:'arm64',electronVersion,overwrite:true,asar:false,prune:false,appBundleId:'org.kikuchicuda.kamiak-di.react',appVersion:'0.1.0'});
 console.log('Native desktop app:',path.join(builds[0],'Kamiak DI.app'));
}finally{await fs.rm(stage,{recursive:true,force:true});}
