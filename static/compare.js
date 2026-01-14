// static/compare.js
// ================= RESUME COMPARISON (ES MODULE) =================

export function initResumeComparison() {
    const compareForm = document.querySelector('form[action="/compare_resumes"]');
    if (!compareForm) return;

    compareForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const formData = new FormData(compareForm);

        try {
            const res = await fetch("/compare_resumes", {
                method: "POST",
                body: formData
            });

            const data = await res.json();

            const html = `
                <div class="glass-card" id="compare_results">
                    <h3>🏆 Better Resume: ${data.winner.replace("_", " ").toUpperCase()}</h3>

                    <table style="width:100%; margin-top:1rem;">
                        <tr>
                            <th></th>
                            <th>Resume 1</th>
                            <th>Resume 2</th>
                        </tr>
                        <tr>
                            <td>Final Score</td>
                            <td>${(data.resume_1.final * 100).toFixed(1)}%</td>
                            <td>${(data.resume_2.final * 100).toFixed(1)}%</td>
                        </tr>
                        <tr>
                            <td>Job Match</td>
                            <td>${(data.resume_1.match * 100).toFixed(1)}%</td>
                            <td>${(data.resume_2.match * 100).toFixed(1)}%</td>
                        </tr>
                        <tr>
                            <td>JD Coverage</td>
                            <td>${data.resume_1.coverage}%</td>
                            <td>${data.resume_2.coverage}%</td>
                        </tr>
                    </table>
                </div>
            `;

            const existing = document.getElementById("compare_results");
            if (existing) existing.remove();

            compareForm.parentElement.insertAdjacentHTML("beforeend", html);

        } catch (err) {
            console.error("Resume comparison failed:", err);
        }
    });
}
