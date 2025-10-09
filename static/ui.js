(function(){
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('image');
  const browseBtn = document.getElementById('browseBtn');
  const preview = document.getElementById('preview');
  const previewImg = document.getElementById('previewImg');
  const fileName = document.getElementById('fileName');
  const fileSize = document.getElementById('fileSize');
  const submitBtn = document.getElementById('submitBtn');
  const thumbs = document.getElementById('thumbs');
  let submitting = false;

  if(!dropzone) return;

  const maxSize = 10 * 1024 * 1024; // 10MB

  function setFile(file){
    if(!file) return;
    if(file.size > maxSize){
      alert('File is too large. Max 10 MB.');
      fileInput.value = '';
      submitBtn.disabled = true;
      return;
    }
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    fileName.textContent = file.name;
    fileSize.textContent = `${(file.size/1024).toFixed(1)} KB`;
    preview.classList.remove('hidden');
    submitBtn.disabled = false;
    renderThumbs([file]);
  }

  browseBtn.addEventListener('click', ()=> fileInput.click());
  dropzone.addEventListener('click', ()=> fileInput.click());
  dropzone.addEventListener('dragover', (e)=>{ e.preventDefault(); dropzone.classList.add('focus'); });
  dropzone.addEventListener('dragleave', ()=> dropzone.classList.remove('focus'));
  dropzone.addEventListener('drop', (e)=>{
    e.preventDefault();
    dropzone.classList.remove('focus');
    if(e.dataTransfer.files && e.dataTransfer.files[0]){
      fileInput.files = e.dataTransfer.files;
      setFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', ()=>{
    if(fileInput.files && fileInput.files[0]) setFile(fileInput.files[0]);
    if(fileInput.files && fileInput.files.length>1){
      renderThumbs(Array.from(fileInput.files));
    }
  });

  // Disable UI while submitting and show spinner text
  const form = document.querySelector('form.uploader');
  if(form){
    form.addEventListener('submit', ()=>{
      if(submitting) return;
      submitting = true;
      submitBtn.disabled = true;
      submitBtn.textContent = 'Predicting…';
      dropzone.classList.add('focus');
    });
  }

  // Thumbnails (front-end queue)
  function renderThumbs(files){
    if(!thumbs) return;
    thumbs.innerHTML = '';
    files.slice(0,6).forEach((f, idx)=>{
      const url = URL.createObjectURL(f);
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'thumb';
      el.title = f.name;
      el.innerHTML = `<img src="${url}" alt="${f.name}">`;
      el.addEventListener('click', ()=>{
        // select this file as active
        fileInput.files = createFileList([f]);
        setFile(f);
      });
      thumbs.appendChild(el);
    });
  }

  function createFileList(files){
    const dt = new DataTransfer();
    files.forEach(f=> dt.items.add(f));
    return dt.files;
  }

  // Keyboard shortcuts
  document.addEventListener('keydown', (e)=>{
    if(e.key === 'Enter' && !submitBtn.disabled){ form.requestSubmit(); }
    if(e.key === 'Escape'){ fileInput.value = ''; submitBtn.disabled = true; preview.classList.add('hidden'); thumbs && (thumbs.innerHTML=''); }
  });

  // Apply saved theme if present (no visible toggle)
  (function(){
    const saved = localStorage.getItem('theme');
    if(saved){ document.documentElement.dataset.theme = saved; }
  })();
})();


