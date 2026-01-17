import { initBatchScreening } from "./batch.js";
import { initResumeComparison } from "./compare.js";
import { initModeSwitch } from "./mode-switch.js";
import { initExportBatchPDF } from "./export-batch-pdf.js";
import { initExportComparePDF } from "./export-compare-pdf.js";

document.addEventListener("DOMContentLoaded", () => {
    initBatchScreening();
    initResumeComparison();
    initModeSwitch();
    initExportBatchPDF();
    initExportComparePDF();
});
