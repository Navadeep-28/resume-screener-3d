import { initModeSwitch } from "./mode-switch.js";
import { initExportBatchPDF } from "./export-batch-pdf.js";
import { initExportComparePDF } from "./export-compare-pdf.js";

document.addEventListener("DOMContentLoaded", () => {
    initModeSwitch();
    initExportBatchPDF();
    initExportComparePDF();
});
