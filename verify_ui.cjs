const { chromium } = require('playwright');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
(async()=>{
 const processes=[];let browser;let log='';
 const env={...process.env, PATHSNAP_MODE:'student',PATHSNAP_API:'http://127.0.0.1:18741'};
 function serve(module,port){
 const p=spawn('.venv/Scripts/python.exe',['-m','uvicorn',module+':app','--port',String(port)],{env,windowsHide:true});
 processes.push(p);for(const s of [p.stdout,p.stderr])s.on('data',d=>{if(module==='api_server')log+=d.toString()});}
 try{
 serve('api_server',18741);serve('ui_server',18742);
 let ids;
 for(let i=0;i<100;i++){try{let r=await fetch('http://127.0.0.1:18742/trips');if(r.ok){ids=await r.json();break}}catch{}await new Promise(r=>setTimeout(r,300));}
 if(ids?.length!==10)throw Error('trip listing failed');
 browser=await chromium.launch({channel:'chrome',headless:true});
 const page=await browser.newPage({viewport:{width:1120,height:850}});
 await page.goto('http://127.0.0.1:18742');
 const expected=fs.readFileSync('experiments/decoder_results.csv','utf8').trim().split(/\r?\n/).slice(1).map(x=>x.split(',')).filter(x=>x[1]==='viterbi');
 for(const id of ids){
 await page.selectOption('#trip',id);
 const response=page.waitForResponse(r=>r.url().endsWith('/predict'));
 await page.click('#run');const data=await (await response).json();
 const row=expected.find(r=>r[0]===id);
 if(!row || data.observations!==Number(row[3]) || Math.round(data.accuracy*data.observations)!==Number(row[4]) || Object.keys(data).some(k=>/label|reference|ground/i.test(k)))throw Error(id);
 await page.waitForFunction(()=>document.querySelector('#status').textContent==='Reconstruction complete');
 if(id===ids[0])await page.screenshot({path:'evidence/student-ui.png'});
 }
 if((await fetch('http://127.0.0.1:18742/trips/99999999/predict',{method:'POST'})).status!==404)throw Error('invalid trip');
 fs.writeFileSync('evidence/api-access.log',log);
 console.log('All ten browser reconstructions matched offline results; unknown trip returned 404.');
 }finally{if(browser)await browser.close();processes.forEach(p=>p.kill())}
})().catch(e=>{console.error(e);process.exitCode=1});
