const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('image-upload-area');
const placeholder = document.getElementById('upload-placeholder');
const mainImage = document.getElementById('main-image');
const identifyBtn = document.getElementById('identify-btn');

let selectedFile = null;

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--accent-green)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--panel-border)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--panel-border)';
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) return;
    
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        mainImage.src = e.target.result;
        mainImage.style.display = 'block';
        placeholder.style.display = 'none';
        
        // Reset UI
        document.getElementById('common-name').textContent = 'Ready for Identification';
        document.getElementById('scientific-name').textContent = '';
        identifyBtn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Identify Arthropod';
        identifyBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

identifyBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    identifyBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
    identifyBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('API Error');

        const data = await response.json();
        updateUI(data);

    } catch (error) {
        console.error(error);
        identifyBtn.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Error';
        document.getElementById('common-name').textContent = 'Failed to connect to backend';
        setTimeout(() => {
            identifyBtn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Identify Arthropod';
            identifyBtn.disabled = false;
        }, 3000);
    }
});

function updateUI(data) {
    identifyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Identified';
    
    // Top Panel
    document.getElementById('common-name').textContent = data.class_name;
    document.getElementById('scientific-name').textContent = data.details.scientific_name;
    document.getElementById('confidence-text').textContent = `Confidence: ${data.confidence}%`;

    
    const hazardStatus = document.getElementById('hazard-status');
    hazardStatus.textContent = data.details.hazard.level === 'Low Hazard' ? 'Safe' : 'Danger';
    hazardStatus.style.backgroundColor = data.details.hazard.level_color;

    // Middle Panel - Crops
    const cropsContainer = document.getElementById('crop-badges');
    cropsContainer.innerHTML = '';
    data.details.impact.crops.forEach(crop => {
        const span = document.createElement('span');
        span.className = 'badge';
        span.textContent = crop;
        cropsContainer.appendChild(span);
    });

    document.getElementById('loss-title').textContent = `Economic loss level: ${data.details.impact.loss_level}`;
    document.getElementById('loss-desc').textContent = data.details.impact.loss_desc;
    
    const actionsList = document.getElementById('actions-list');
    actionsList.innerHTML = '';
    data.details.impact.actions.forEach(action => {
        const li = document.createElement('li');
        li.textContent = action;
        actionsList.appendChild(li);
    });

    populateList('organic-list', data.details.impact.organic);
    populateList('chemical-list', data.details.impact.chemical);
    populateList('biological-list', data.details.impact.biological);

    // Middle Panel - Hazard
    const humanHazardLevel = document.getElementById('human-hazard-level');
    humanHazardLevel.textContent = data.details.hazard.level;
    humanHazardLevel.style.backgroundColor = data.details.hazard.level_color;

    const traitsContainer = document.getElementById('traits-list');
    traitsContainer.innerHTML = '';
    for (const [trait, active] of Object.entries(data.details.hazard.traits)) {
        const span = document.createElement('span');
        span.className = `trait ${active ? 'active' : 'inactive'}`;
        span.textContent = `${active ? '✓' : 'X'} ${trait}`;
        traitsContainer.appendChild(span);
    }

    document.getElementById('first-aid-text').textContent = data.details.hazard.first_aid;
    document.getElementById('precautions-text').textContent = data.details.hazard.precautions;

    // Bottom Panel - Explainability
    document.getElementById('orig-explain-img').src = data.original_base64;
    document.getElementById('heatmap-img').src = data.heatmap_base64;
    
    const infLevel = document.getElementById('infestation-level');
    infLevel.textContent = data.details.impact.infestation_level;
    infLevel.style.backgroundColor = data.details.impact.loss_level === 'High' ? 'var(--accent-red)' : 'var(--accent-orange)';
    
    document.getElementById('heatmap-desc').textContent = data.details.impact.heatmap_desc;
}

function populateList(elementId, items) {
    const list = document.getElementById(elementId);
    list.innerHTML = '';
    items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        list.appendChild(li);
    });
}
