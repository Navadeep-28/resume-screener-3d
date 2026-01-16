// static/batch.js
// ================= BATCH RESUME SCREENING (ES MODULE) =================

export function initBatchScreening() {
    const batchForm = document.querySelector('form[action="/batch-screen"]');
    if (!batchForm) return;

    batchForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const formData = new FormData(batchForm);

        try {
            const res = await fetch("/batch-screen", {
                method: "POST",
                body: formData
            });

            // ---- SAFETY CHECK: JSON vs HTML ----
            const contentType = res.headers.get("content-type") || "";

            if (!contentType.includes("application/json")) {
                const text = await res.text();
                console.error("Expected JSON but received:", text);
                throw new Error("Server returned HTML instead of JSON");
            }

            const data = await res.json();

            if (!data.ranked_results || !Array.isArray(data.ranked_results)) {
                throw new Error("Invalid batch response format");
            }

            // ---- RENDER RESULTS (SINGLE-STYLE UI) ----
            const container = document.getElementById("dynamic-results");
            if (!container) return;

            container.innerHTML = ""; // clear old results

            data.ranked_results.forEach((r, index) => {
                const status = r.final >= 65 ? "PASS" : "FAIL";
                const risk = r.final < 60 ? "HIGH" : "LOW";

                container.innerHTML += `
                <section class="results glass-card result-landscape">
                    <div class="result-col status-col">
                        <div class="status-pill ${status === "PASS" ? "pass" : "fail"}">
                            ${status}
                        </div>
                        <div class="status-card">
                            <h3>#${index + 1} ${r.filename}</h3>
                            <p>ATS Confidence: 70%</p>
                        </div>
                    </div>

                    <div class="result-col metrics-col">
                        <div class="metric-card">
                            <span class="metric-label">JD Coverage</span>
                            <span class="metric-value blue">${r.coverage}%</span>
                        </div>
                        <div class="metric-card">
                            <span class="metric-label">Hiring Risk</span>
                            <span class="metric-value ${risk === "HIGH" ? "danger" : "success"}">
                                ${risk}
                            </span>
                        </div>
                    </div>

                    <div class="result-col score-col">
                        <div class="final-score">
                            <span>Final Score</span>
                            <h2>${r.final.toFixed(1)}%</h2>
                        </div>
                        <div class="mini-scores">
                            <div class="mini-card">
                                Quality<br><strong>${r.quality}%</strong>
                            </div>
                            <div class="mini-card">
                                Job Match<br><strong>${r.match}%</strong>
                            </div>
                        </div>
                    </div>
                </section>
                `;
            });

        } catch (err) {
            console.error("Batch screening failed:", err);
        }
    });
}
