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

  // Clean up object URLs to prevent memory leaks
  const objectURLs = new Set();
  
  function cleanupObjectURLs() {
    objectURLs.forEach(url => {
      URL.revokeObjectURL(url);
    });
    objectURLs.clear();
  }
  
  function showError(message, type = 'error') {
    // Remove existing error messages
    const existingError = document.querySelector('.error-message');
    if (existingError) {
      existingError.remove();
    }
    
    const errorDiv = document.createElement('div');
    errorDiv.className = `error-message ${type}`;
    errorDiv.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
      </svg>
      <span>${message}</span>
    `;
    
    // Insert after dropzone
    dropzone.parentNode.insertBefore(errorDiv, dropzone.nextSibling);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
      if (errorDiv.parentNode) {
        errorDiv.remove();
      }
    }, 5000);
  }
  
  function setFile(file){
    if(!file) return;
    
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      showError('Please select a valid image file (JPG, PNG, or WebP)');
      fileInput.value = '';
      submitBtn.disabled = true;
      return;
    }
    
    if(file.size > maxSize){
      showError('File is too large. Maximum size is 10 MB.');
      fileInput.value = '';
      submitBtn.disabled = true;
      return;
    }
    
    // Clean up previous object URLs
    cleanupObjectURLs();
    
    try {
      const url = URL.createObjectURL(file);
      objectURLs.add(url);
      
      previewImg.src = url;
      fileName.textContent = file.name;
      fileSize.textContent = `${(file.size/1024).toFixed(1)} KB`;
      preview.classList.remove('hidden');
      submitBtn.disabled = false;
      renderThumbs([file]);
    } catch (error) {
      console.error('Error processing file:', error);
      showError('Error processing the selected file. Please try again.');
      fileInput.value = '';
      submitBtn.disabled = true;
    }
  }

  browseBtn.addEventListener('click', (e)=> {
    e.preventDefault();
    e.stopPropagation();
    fileInput.click();
  });
  dropzone.addEventListener('click', (e)=> {
    e.preventDefault();
    fileInput.click();
  });
  dropzone.addEventListener('dragover', (e)=>{ 
    e.preventDefault(); 
    e.stopPropagation();
    dropzone.classList.add('focus'); 
  });
  dropzone.addEventListener('dragleave', (e)=> {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove('focus'); 
  });
  dropzone.addEventListener('drop', (e)=>{
    e.preventDefault();
    e.stopPropagation();
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
    form.addEventListener('submit', (e)=>{
      if(submitting) {
        e.preventDefault();
        return;
      }
      
      // Validate file before submission
      if (!fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        showError('Please select an image file first.');
        return;
      }
      
      submitting = true;
      submitBtn.disabled = true;
      submitBtn.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="spinning">
          <circle cx="12" cy="12" r="10"></circle>
          <path d="M12 6v6l4 2"></path>
        </svg>
        <span>Predicting…</span>
      `;
      dropzone.classList.add('focus');
    });
  }

  // Thumbnails (front-end queue)
  function renderThumbs(files){
    if(!thumbs) return;
    
    // Clean up existing thumbnails
    thumbs.innerHTML = '';
    
    files.slice(0,6).forEach((f, idx)=>{
      const url = URL.createObjectURL(f);
      objectURLs.add(url);
      
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'thumb';
      el.title = f.name;
      el.setAttribute('aria-label', `Select ${f.name}`);
      
      const img = document.createElement('img');
      img.src = url;
      img.alt = f.name;
      img.loading = 'lazy'; // Lazy loading for performance
      
      el.appendChild(img);
      
      el.addEventListener('click', (e)=>{
        e.preventDefault();
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
    if(e.key === 'Enter' && !submitBtn.disabled && !submitting){ 
      e.preventDefault();
      form.requestSubmit(); 
    }
    if(e.key === 'Escape'){ 
      fileInput.value = ''; 
      submitBtn.disabled = true; 
      preview.classList.add('hidden'); 
      thumbs && (thumbs.innerHTML=''); 
    }
  });

  // Apply saved theme if present (no visible toggle)
  (function(){
    const saved = localStorage.getItem('theme');
    if(saved){ document.documentElement.dataset.theme = saved; }
  })();
  
  // Cleanup on page unload
  window.addEventListener('beforeunload', cleanupObjectURLs);
  
  // Cleanup on form reset
  if(form){
    form.addEventListener('reset', cleanupObjectURLs);
  }
})();


