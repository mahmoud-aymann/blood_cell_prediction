(function(){
  // Zoom/Pan controls
  const img = document.getElementById('zoomImg');
  const zoomIn = document.getElementById('zoomIn');
  const zoomOut = document.getElementById('zoomOut');
  const zoomReset = document.getElementById('zoomReset');
  let scale = 1;
  let originX = 0, originY = 0;
  let isPanning = false; let startX=0, startY=0;

  function applyTransform(){
    img.style.transform = `translate(${originX}px, ${originY}px) scale(${scale})`;
    img.style.transformOrigin = 'center center';
  }

  zoomIn && zoomIn.addEventListener('click', ()=>{ scale = Math.min(scale*1.2, 8); applyTransform(); });
  zoomOut && zoomOut.addEventListener('click', ()=>{ scale = Math.max(scale/1.2, 0.25); applyTransform(); });
  zoomReset && zoomReset.addEventListener('click', ()=>{ scale = 1; originX=0; originY=0; applyTransform(); });

  // Pan with mouse drag
  if(img){
    img.style.cursor = 'grab';
    img.addEventListener('mousedown', (e)=>{ isPanning = true; startX = e.clientX - originX; startY = e.clientY - originY; img.style.cursor='grabbing'; });
    document.addEventListener('mouseup', ()=>{ isPanning=false; img.style.cursor='grab'; });
    document.addEventListener('mousemove', (e)=>{ if(!isPanning) return; originX = e.clientX - startX; originY = e.clientY - startY; applyTransform(); });
    img.addEventListener('wheel', (e)=>{ e.preventDefault(); const delta = e.deltaY>0 ? 0.9 : 1.1; scale = Math.min(Math.max(scale*delta, 0.25), 8); applyTransform(); }, { passive:false });
  }

  // Copy result (top prediction and confidence)
  const copyBtn = document.getElementById('copyResult');
  if(copyBtn){
    copyBtn.addEventListener('click', ()=>{
      const title = document.querySelector('.card h2');
      const conf = document.querySelector('.card p');
      const text = `${title ? title.textContent : 'Prediction'} - ${conf ? conf.textContent : ''}`;
      navigator.clipboard.writeText(text).then(()=>{ copyBtn.textContent='Copied!'; setTimeout(()=> copyBtn.textContent='Copy result', 1200); });
    });
  }

  // Print / PDF
  const printBtn = document.getElementById('printResult');
  printBtn && printBtn.addEventListener('click', ()=> window.print());
})();


