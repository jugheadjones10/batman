// Interactive Command Builder for Batman Documentation
(function () {
  "use strict";

  // Initialize all command builders on page load
  document.addEventListener("DOMContentLoaded", function () {
    const builders = document.querySelectorAll(".command-builder-widget");
    builders.forEach((element) => {
      const toolName = element.dataset.tool;
      const params = JSON.parse(element.dataset.params || "[]");
      new CommandBuilder(element, toolName, params);
    });
  });

  class CommandBuilder {
    constructor(container, toolName, parameters) {
      this.container = container;
      this.toolName = toolName;
      this.parameters = parameters;
      this.values = {};

      // Load saved values from localStorage
      this.loadValues();

      // Render the form
      this.render();
    }

    loadValues() {
      const saved = localStorage.getItem(`batman_${this.toolName}`);
      if (saved) {
        try {
          this.values = JSON.parse(saved);
        } catch (e) {
          console.error("Failed to load saved values:", e);
        }
      }
    }

    saveValues() {
      localStorage.setItem(
        `batman_${this.toolName}`,
        JSON.stringify(this.values),
      );
    }

    render() {
      // Group parameters
      const groups = {};
      this.parameters.forEach((param) => {
        const group = param.group || "General";
        if (!groups[group]) groups[group] = [];
        groups[group].push(param);
      });

      // Build HTML
      let html = '<div class="command-builder">';

      // Form groups
      html += '<div class="command-builder-form">';
      Object.entries(groups).forEach(([groupName, params]) => {
        html += `<div class="param-group">`;
        html += `<h4>${groupName}</h4>`;
        params.forEach((param) => {
          html += this.renderParameter(param);
        });
        html += "</div>";
      });
      html += "</div>";

      // Command output
      html += '<div class="command-builder-output">';
      html += '<div class="command-output-header">';
      html += "<h4>Generated Command</h4>";
      html += '<button id="copy-command" class="copy-btn">Copy</button>';
      html += '<button id="reset-command" class="reset-btn">Reset</button>';
      html += "</div>";
      html += '<pre id="command-output"></pre>';
      html += "</div>";

      html += "</div>";

      this.container.innerHTML = html;

      // Attach event listeners
      this.attachListeners();

      // Generate initial command
      this.updateCommand();
    }

    renderParameter(param) {
      const value = this.values[param.name] ?? param.default ?? "";
      const required = param.required ? ' <span class="required">*</span>' : "";

      let html = `<div class="param-field" data-param="${param.name}">`;
      html += `<label for="param_${param.name}">${this.formatLabel(param.name)}${required}</label>`;

      if (param.description) {
        html += `<span class="param-description">${param.description}</span>`;
      }

      switch (param.type) {
        case "choice":
          html += `<select id="param_${param.name}" name="${param.name}">`;
          param.choices.forEach((choice) => {
            const selected = value === choice ? "selected" : "";
            html += `<option value="${choice}" ${selected}>${choice}</option>`;
          });
          html += "</select>";
          break;

        case "flag":
          const checked = value ? "checked" : "";
          html += `<input type="checkbox" id="param_${param.name}" name="${param.name}" ${checked}>`;
          break;

        case "number":
          html += `<input type="number" id="param_${param.name}" name="${param.name}" value="${value}"`;
          if (param.min !== undefined) html += ` min="${param.min}"`;
          if (param.max !== undefined) html += ` max="${param.max}"`;
          if (param.step !== undefined) html += ` step="${param.step}"`;
          html += ">";
          break;

        default: // text, path
          html += `<input type="text" id="param_${param.name}" name="${param.name}" value="${value}" placeholder="${param.description || ""}">`;
      }

      html += "</div>";
      return html;
    }

    formatLabel(name) {
      // Convert parameter name to readable label
      return name
        .split("-")
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(" ");
    }

    attachListeners() {
      // Input change listeners
      this.parameters.forEach((param) => {
        const element = document.getElementById(`param_${param.name}`);
        if (element) {
          element.addEventListener("change", () => {
            if (param.type === "flag") {
              this.values[param.name] = element.checked;
            } else {
              this.values[param.name] = element.value;
            }
            this.saveValues();
            this.updateCommand();
          });

          if (param.type !== "choice" && param.type !== "flag") {
            element.addEventListener("input", () => {
              this.values[param.name] = element.value;
              this.updateCommand();
            });
          }
        }
      });

      // Copy button
      const copyBtn = document.getElementById("copy-command");
      if (copyBtn) {
        copyBtn.addEventListener("click", () => this.copyCommand());
      }

      // Reset button
      const resetBtn = document.getElementById("reset-command");
      if (resetBtn) {
        resetBtn.addEventListener("click", () => this.resetForm());
      }
    }

    updateCommand() {
      const output = document.getElementById("command-output");
      if (!output) return;

      const command = this.generateCommand();
      output.textContent = command;
    }

    generateCommand() {
      let parts = [];

      // Determine base command
      if (this.toolName.startsWith("submit_")) {
        parts.push(`./${this.toolName}.sh`);
      } else {
        parts.push("python -m cli." + this.toolName);
      }

      // Add parameters
      this.parameters.forEach((param) => {
        const value = this.values[param.name];

        if (param.type === "flag") {
          if (value) {
            parts.push(`--${param.name}`);
          }
        } else if (
          value !== undefined &&
          value !== "" &&
          value !== param.default
        ) {
          // Only add if different from default
          if (param.type === "path" && value.includes(" ")) {
            parts.push(`--${param.name}="${value}"`);
          } else {
            parts.push(`--${param.name}=${value}`);
          }
        } else if (param.required && (value === undefined || value === "")) {
          // Show placeholder for required params
          parts.push(`--${param.name}=<REQUIRED>`);
        }
      });

      // Format with line continuations for readability
      if (parts.length > 3) {
        return parts[0] + " \\\n  " + parts.slice(1).join(" \\\n  ");
      }
      return parts.join(" ");
    }

    copyCommand() {
      const output = document.getElementById("command-output");
      if (!output) return;

      const text = output.textContent;
      navigator.clipboard
        .writeText(text)
        .then(() => {
          const btn = document.getElementById("copy-command");
          const originalText = btn.textContent;
          btn.textContent = "Copied!";
          setTimeout(() => {
            btn.textContent = originalText;
          }, 2000);
        })
        .catch((err) => {
          console.error("Failed to copy:", err);
          alert("Failed to copy to clipboard");
        });
    }

    resetForm() {
      this.values = {};
      this.saveValues();

      // Reset all form elements
      this.parameters.forEach((param) => {
        const element = document.getElementById(`param_${param.name}`);
        if (!element) return;

        if (param.type === "flag") {
          element.checked = false;
        } else if (param.type === "choice") {
          element.value = param.default || param.choices[0];
        } else {
          element.value = param.default || "";
        }
      });

      this.updateCommand();
    }
  }
})();
