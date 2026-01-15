// static/mode-switch.js
// ================= MODE SWITCH (FIXED ES MODULE) =================

export function initModeSwitch() {
    const buttons = document.querySelectorAll(".mode-btn");
    const sections = document.querySelectorAll(".mode-section");

    if (buttons.length === 0 || sections.length === 0) return;

    function switchMode(mode) {
        // Toggle sections
        sections.forEach(section => {
            section.style.display =
                section.dataset.mode === mode ? "block" : "none";
        });

        // Toggle active button
        buttons.forEach(btn => {
            btn.classList.toggle("active", btn.dataset.mode === mode);
        });
    }

    // Default mode
    switchMode("single");

    // Button clicks
    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            switchMode(btn.dataset.mode);
        });
    });
}
