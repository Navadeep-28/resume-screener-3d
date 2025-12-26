// ========== 3D BACKGROUND SETUP ==========
let scene, camera, renderer, stars, ringMesh, clock;

function initThreeJS() {
    const canvas = document.getElementById('three-canvas');

    if (!canvas || typeof THREE === 'undefined') {
        console.warn('Three.js or canvas not available, skipping 3D background');
        return;
    }

    scene = new THREE.Scene();
    clock = new THREE.Clock();

    camera = new THREE.PerspectiveCamera(
        70,
        window.innerWidth / window.innerHeight,
        0.1,
        1500
    );
    camera.position.set(0, 0, 140);

    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    canvas.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0x442266, 0.8);
    scene.add(ambient);

    const pointLight = new THREE.PointLight(0x00e6ff, 1.2, 600);
    pointLight.position.set(0, 80, 120);
    scene.add(pointLight);

    const particles = 800;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particles * 3);

    for (let i = 0; i < positions.length; i += 3) {
        positions[i]     = (Math.random() - 0.5) * 2000;
        positions[i + 1] = (Math.random() - 0.5) * 1200;
        positions[i + 2] = (Math.random() - 0.5) * 2000;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
        color: 0x00e6ff,
        size: 3,
        transparent: true,
        opacity: 0.8
    });
    stars = new THREE.Points(geometry, material);
    scene.add(stars);

    const ringGeom = new THREE.TorusBufferGeometry(50, 4, 32, 120);
    const ringMat = new THREE.MeshStandardMaterial({
        color: 0xff00ff,
        emissive: 0x550055,
        metalness: 0.7,
        roughness: 0.25
    });
    ringMesh = new THREE.Mesh(ringGeom, ringMat);
    ringMesh.rotation.x = Math.PI / 3;
    scene.add(ringMesh);

    animate();
}

function animate() {
    if (!renderer || !scene || !camera) return;

    requestAnimationFrame(animate);

    const t = clock ? clock.getElapsedTime() : performance.now() * 0.001;

    if (stars) {
        stars.rotation.x += 0.00025;
        stars.rotation.y += 0.0004;
    }

    if (ringMesh) {
        ringMesh.rotation.y += 0.004;
        ringMesh.rotation.z = Math.sin(t * 0.6) * 0.4;
    }

    const radius = 140;
    camera.position.x = Math.cos(t * 0.1) * radius;
    camera.position.z = Math.sin(t * 0.1) * radius;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    if (!camera || !renderer) return;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ========== FORM & ANALYSIS LOGIC ==========
document.addEventListener('DOMContentLoaded', function () {
    initThreeJS();

    const form    = document.getElementById('uploadForm');
    const loading = document.getElementById('loading');
    const resultsContainer = document.getElementById('resultsContainer');

    if (!form) return;

    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        if (loading) loading.classList.remove('d-none');
        if (resultsContainer) resultsContainer.innerHTML = '';

        const formData = new FormData(form);

        try {
            const response = await fetch(form.action || '/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!data || data.error) {
                alert('Error: ' + (data.error || 'Unknown error'));
                return;
            }

            // Decide which mode based on backend response
            if (data.compare_results) {
                renderCompareResults(data, resultsContainer);
            } else if (data.batch_results) {
                renderBatchResults(data, resultsContainer);
            } else if (data.results) {
                renderSingleResult(data, resultsContainer);
            } else {
                alert('Unexpected response from server.');
            }
        } catch (error) {
            alert('Error analyzing resume: ' + error.message);
        } finally {
            if (loading) loading.classList.add('d-none');
        }
    });
});

// ---------- SINGLE RESULT (Idea 1,2,6) ----------
function renderSingleResult(data, container) {
    const results = data.results;
    const pdfUrl = data.pdf_url;

    const overall = Number(results.overall_score || results.overall || 0).toFixed(1);
    const tfidf = Number(results.tfidf_similarity || results.tfidf || 0).toFixed(1);
    const skills = Number(results.skills_match || results.skills || 0).toFixed(1);
    const exp = Number(results.experience || 0).toFixed(1);
    const jdMode = !!results.jd_mode;
    const jdMatch = Number(results.jd_keyword_match || 0).toFixed(1);

    const blindUsed = results.blind_mode;

    // Build HTML block
    const html = `
      <div class="card glass-card shadow-lg">
        <div class="card-header bg-gradient-primary text-white">
          <h3 class="mb-0">
            <i class="fas fa-chart-line me-2"></i>Analysis Complete
          </h3>
        </div>
        <div class="card-body p-4">
          <div class="row text-center mb-4">
            <div class="col-12">
              <div class="score-circle mb-3" id="overallScore">${overall}%</div>
              <h4 class="text-white" id="jobTitle">${results.job_category || ''}</h4>
              <p class="text-white-50 small" id="decisionText">
                Overall decision based on TF‑IDF match, skills, and experience.
              </p>
              ${blindUsed ? '<p class="text-info small mb-0">Blind screening: identity fields were hidden from the model.</p>' : ''}
              ${jdMode ? '<p class="text-info small mb-0">Custom job description was used for matching.</p>' : ''}
            </div>
          </div>

          <div class="row">
            <div class="col-md-6 mb-3">
              <div id="scorePie" style="width:100%; height:250px;"></div>
            </div>
            <div class="col-md-6 mb-3">
              <div id="scoreBars" style="width:100%; height:250px;"></div>
            </div>
          </div>

          <div class="row mt-3">
            <div class="col-12">
              <div id="overlapHeatmap" style="width:100%; height:300px;"></div>
            </div>
          </div>

          <div class="mt-4">
            <h5 class="text-white mb-3">
              <i class="fas fa-question-circle me-2"></i>Why this score?
            </h5>
            <div class="row">
              <div class="col-md-4 mb-3">
                <div class="text-white-50 small">
                  <div class="fw-bold text-white mb-1">Top matching phrases</div>
                  <ul id="explainTopPhrases" class="small mb-0"></ul>
                </div>
              </div>
              <div class="col-md-4 mb-3">
                <div class="text-white-50 small">
                  <div class="fw-bold text-white mb-1">Missing critical skills</div>
                  <ul id="explainMissingCritical" class="small mb-0"></ul>
                </div>
              </div>
              <div class="col-md-4 mb-3">
                <div class="text-white-50 small">
                  <div class="fw-bold text-white mb-1">Experience rationale</div>
                  <p id="explainExperience" class="small mb-0"></p>
                </div>
              </div>
            </div>
          </div>

          <div class="mt-4">
            <a id="downloadBtn" class="btn btn-success btn-lg w-100" href="${pdfUrl || '#'}" download>
              <i class="fas fa-file-pdf me-2"></i>Download Detailed PDF Report
            </a>
          </div>

          <div class="mt-4 text-white-50 small" id="skillsSummary"></div>

          ${jdMode ? `
          <div class="mt-4 text-white-50 small">
            <b>JD keyword match:</b> ${jdMatch}%<br>
            <b>JD keywords found:</b> ${(results.jd_used_keywords || []).join(', ') || 'None'}<br>
            <b>JD keywords missing:</b> ${(results.jd_missing_keywords || []).join(', ') || 'None'}
          </div>` : ''}
        </div>
      </div>
    `;

    container.innerHTML = html;

    // After injecting HTML, hook up details + charts
    applySingleResultDetails(results);
}

function applySingleResultDetails(results) {
    const scoreCircle   = document.getElementById('overallScore');
    const decisionText  = document.getElementById('decisionText');
    const skillsSummary = document.getElementById('skillsSummary');

    const overall = Number(results.overall_score || results.overall || 0);
    if (scoreCircle) {
        scoreCircle.textContent = overall.toFixed(1) + '%';
        const accepted = overall >= 70;
        scoreCircle.style.background = accepted
            ? 'radial-gradient(circle, #3cff9c 0%, #00ffd0 40%, #006f52 100%)'
            : 'radial-gradient(circle, #ff6b6b 0%, #ff2e63 40%, #7f0000 100%)';
        if (decisionText) {
            decisionText.textContent = accepted
                ? 'This resume meets the threshold for this role.'
                : 'This resume currently scores below the threshold for this role.';
        }
    }

    if (skillsSummary) {
        const matched = (results.matched_skills || []).join(', ') || 'None';
        const missing = (results.missing_skills || []).join(', ') || 'None';
        skillsSummary.innerHTML =
            `<b>Matched skills:</b> ${matched}<br>` +
            `<b>Missing skills:</b> ${missing}`;
    }

    const explanation = results.explanation || {};
    const explainTop   = document.getElementById('explainTopPhrases');
    const explainMiss  = document.getElementById('explainMissingCritical');
    const explainExp   = document.getElementById('explainExperience');

    if (explainTop) {
        explainTop.innerHTML = '';
        (explanation.top_matching_phrases || []).forEach(p => {
            const li = document.createElement('li');
            li.textContent = p;
            explainTop.appendChild(li);
        });
        if (!explanation.top_matching_phrases || !explanation.top_matching_phrases.length) {
            explainTop.innerHTML = '<li>No strong matches detected.</li>';
        }
    }

    if (explainMiss) {
        explainMiss.innerHTML = '';
        (explanation.missing_critical_skills || []).forEach(s => {
            const li = document.createElement('li');
            li.textContent = s;
            explainMiss.appendChild(li);
        });
        if (!explanation.missing_critical_skills || !explanation.missing_critical_skills.length) {
            explainMiss.innerHTML = '<li>No critical gaps found.</li>';
        }
    }

    if (explainExp) {
        explainExp.textContent = explanation.experience_explanation ||
            'Experience score is based on extracted years and seniority terms in the resume.';
    }

    createScoreCharts(results);
}

// ---------- BATCH RESULTS (Idea 4) ----------
function renderBatchResults(data, container) {
    const results = data.batch_results || [];
    const batchId = data.batch_id;
    const total = data.total_processed || results.length;

    if (!results.length) {
        container.innerHTML = '<p class="text-white-50">No valid resumes processed.</p>';
        return;
    }

    const rows = results.map((r, idx) => {
        const score = Number(r.overall_score || r.overall || 0).toFixed(1);
        const status = score >= 70 ? '✅ Shortlist' : 'ℹ️ Review';
        return `
          <tr>
            <td>${idx + 1}</td>
            <td>${r.filename || 'candidate_' + (idx + 1)}</td>
            <td>${score}%</td>
            <td>${status}</td>
          </tr>`;
    }).join('');

    container.innerHTML = `
      <div class="card glass-card shadow-lg">
        <div class="card-header bg-gradient-success text-white d-flex justify-content-between align-items-center">
          <h3 class="mb-0">
            <i class="fas fa-layer-group me-2"></i>Batch Analysis Complete
          </h3>
          <span class="badge bg-light text-dark fs-6">${total}</span>
        </div>
        <div class="card-body p-4">
          <div class="table-responsive mb-4">
            <table class="table table-dark table-hover table-sm mb-0">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Candidate</th>
                  <th>Score</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                ${rows}
              </tbody>
            </table>
          </div>
          ${batchId ? `
          <a class="btn btn-success btn-lg w-100" href="/download-batch/${batchId}">
            <i class="fas fa-file-archive me-2"></i>Download All Reports (ZIP)
          </a>` : ''}
        </div>
      </div>
    `;
}

// ---------- COMPARE RESULTS (Idea 5) ----------
function renderCompareResults(data, container) {
    const [r1, r2] = data.compare_results;
    const winner = data.winner;
    const s1 = Number(r1.overall_score || r1.overall || 0).toFixed(1);
    const s2 = Number(r2.overall_score || r2.overall || 0).toFixed(1);

    container.innerHTML = `
      <div class="card glass-card shadow-lg">
        <div class="card-header bg-gradient-warning text-white d-flex justify-content-between">
          <h3 class="mb-0">
            <i class="fas fa-balance-scale me-2"></i>Head-to-Head Comparison
          </h3>
          <span class="badge bg-success fs-6">
            Winner: Candidate ${winner}
          </span>
        </div>
        <div class="card-body p-4">
          <div class="row text-center mb-4">
            <div class="col-md-6">
              <div class="score-circle mb-3 ${winner === 1 ? 'border border-success' : ''}">
                ${s1}%
              </div>
              <h5 class="text-white fw-bold">${r1.filename || 'Candidate 1'}</h5>
              <p class="text-white-50 small">
                Top skills: ${(r1.matched_skills || []).slice(0,5).join(', ') || 'N/A'}
              </p>
              <a class="btn btn-outline-primary btn-sm w-100" href="${data.pdf1_url}" download>
                <i class="fas fa-file-pdf me-1"></i>Download Report
              </a>
            </div>
            <div class="col-md-6">
              <div class="score-circle mb-3 ${winner === 2 ? 'border border-success' : ''}">
                ${s2}%
              </div>
              <h5 class="text-white fw-bold">${r2.filename || 'Candidate 2'}</h5>
              <p class="text-white-50 small">
                Top skills: ${(r2.matched_skills || []).slice(0,5).join(', ') || 'N/A'}
              </p>
              <a class="btn btn-outline-primary btn-sm w-100" href="${data.pdf2_url}" download>
                <i class="fas fa-file-pdf me-1"></i>Download Report
              </a>
            </div>
          </div>
          <div class="row">
            <div class="col-md-6 mb-3">
              <div id="compareBarChart" style="width:100%; height:300px;"></div>
            </div>
            <div class="col-md-6 mb-3">
              <div id="compareRadarChart" style="width:100%; height:300px;"></div>
            </div>
          </div>
        </div>
      </div>
    `;

    createCompareCharts(r1, r2);
}

// ========== PLOTLY CHARTS WITH NEON THEME ==========
function neonLayout(title) {
    return {
        title: { text: title, font: { color: '#ffffff', size: 16 } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e5e5ff' },
        margin: { l: 40, r: 20, t: 40, b: 40 }
    };
}

function createScoreCharts(results) {
    const pieContainer  = document.getElementById('scorePie');
    const barContainer  = document.getElementById('scoreBars');
    const heatContainer = document.getElementById('overlapHeatmap');

    if (typeof Plotly === 'undefined') {
        console.warn('Plotly not available – charts will not render');
        return;
    }

    const tfidfScore = Number(results.tfidf || results.tfidf_similarity || 0);
    const skillScore = Number(results.skills || results.skills_match || 0);
    const expScore   = Number(results.experience || 0);

    if (pieContainer) {
        const pieData = [{
            type: 'pie',
            values: [tfidfScore, skillScore, expScore],
            labels: ['TF‑IDF match', 'Skills match', 'Experience'],
            hole: 0.45,
            marker: {
                colors: ['#00e6ff', '#ff00ff', '#feca57'],
                line: { color: '#05010a', width: 2 }
            },
            textinfo: 'label+percent',
            hoverinfo: 'label+value+percent',
            pull: [0.02, 0.04, 0.02]
        }];

        const layout = neonLayout('Score components');
        Plotly.newPlot(pieContainer, pieData, layout, { displayModeBar: false });
    }

    if (barContainer) {
        const barData = [{
            x: ['TF‑IDF', 'Skills', 'Experience'],
            y: [tfidfScore, skillScore, expScore],
            type: 'bar',
            marker: {
                color: ['#00e6ff', '#ff00ff', '#3cff9c'],
                line: { color: '#ffffff', width: 1 }
            },
            text: [tfidfScore, skillScore, expScore].map(v => v.toFixed(1) + '%'),
            textposition: 'outside',
            hoverinfo: 'x+y'
        }];

        const layout = neonLayout('Scores (%)');
        layout.yaxis = { range: [0, 100], gridcolor: 'rgba(255,255,255,0.1)' };
        layout.xaxis = { gridcolor: 'rgba(255,255,255,0.05)' };

        Plotly.newPlot(barContainer, barData, layout, { displayModeBar: false });
    }

    if (heatContainer && results.heatmap_data) {
        const h = results.heatmap_data;
        const common       = Number(h.common_words || 0);
        const resumeUnique = Number(h.resume_unique || 0);
        const jdUnique     = Number(h.jd_unique || 0);
        const commonPct    = Number(h.common_percentage || 0);

        const heatData = [{
            z: [[common, resumeUnique, jdUnique]],
            x: ['Common', 'Resume‑only', 'JD‑only'],
            y: ['Overlap'],
            type: 'heatmap',
            colorscale: [
                [0.0, '#1b0034'],
                [0.3, '#00e6ff'],
                [0.6, '#ff00ff'],
                [1.0, '#ff6b6b']
            ],
            showscale: true,
            hoverongaps: false
        }];

        const layout = neonLayout(`Word overlap (common: ${commonPct.toFixed(1)}%)`);
        layout.yaxis = { showgrid: false };
        layout.xaxis = { showgrid: false };

        Plotly.newPlot(heatContainer, heatData, layout, { displayModeBar: false });
    }
}

// ---------- COMPARE CHARTS (Idea 5) ----------
function createCompareCharts(r1, r2) {
    if (typeof Plotly === 'undefined') return;

    const barContainer  = document.getElementById('compareBarChart');
    const radarContainer = document.getElementById('compareRadarChart');

    const s1 = {
        overall: Number(r1.overall_score || r1.overall || 0),
        tfidf: Number(r1.tfidf || r1.tfidf_similarity || 0),
        skills: Number(r1.skills || r1.skills_match || 0),
        exp: Number(r1.experience || 0)
    };
    const s2 = {
        overall: Number(r2.overall_score || r2.overall || 0),
        tfidf: Number(r2.tfidf || r2.tfidf_similarity || 0),
        skills: Number(r2.skills || r2.skills_match || 0),
        exp: Number(r2.experience || 0)
    };

    if (barContainer) {
        const data = [
            {
                x: ['Overall', 'TF‑IDF', 'Skills', 'Experience'],
                y: [s1.overall, s1.tfidf, s1.skills, s1.exp],
                name: r1.filename || 'Candidate 1',
                type: 'bar'
            },
            {
                x: ['Overall', 'TF‑IDF', 'Skills', 'Experience'],
                y: [s2.overall, s2.tfidf, s2.skills, s2.exp],
                name: r2.filename || 'Candidate 2',
                type: 'bar'
            }
        ];
        const layout = neonLayout('Score comparison');
        layout.barmode = 'group';
        layout.yaxis = { range: [0, 100] };
        Plotly.newPlot(barContainer, data, layout, { displayModeBar: false });
    }

    if (radarContainer) {
        const categories = ['Overall', 'TF‑IDF', 'Skills', 'Experience'];
        const data = [
            {
                type: 'scatterpolar',
                r: [s1.overall, s1.tfidf, s1.skills, s1.exp, s1.overall],
                theta: [...categories, categories[0]],
                fill: 'toself',
                name: r1.filename || 'Candidate 1'
            },
            {
                type: 'scatterpolar',
                r: [s2.overall, s2.tfidf, s2.skills, s2.exp, s2.overall],
                theta: [...categories, categories[0]],
                fill: 'toself',
                name: r2.filename || 'Candidate 2'
            }
        ];
        const layout = neonLayout('Radar comparison');
        layout.polar = {
            radialaxis: { visible: true, range: [0, 100] }
        };
        Plotly.newPlot(radarContainer, data, layout, { displayModeBar: false });
    }
}
