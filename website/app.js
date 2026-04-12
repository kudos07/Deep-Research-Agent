(function () {
  "use strict";

  const nav = document.querySelector(".site-nav__links");
  const toggle = document.querySelector(".nav-toggle");

  if (toggle && nav) {
    toggle.addEventListener("click", function () {
      var open = nav.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    nav.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function () {
        nav.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  var sections = Array.prototype.slice.call(document.querySelectorAll("section[id]"));
  var links = Array.prototype.slice.call(document.querySelectorAll('.site-nav__links a[href^="#"]'));

  function setActive() {
    var y = window.scrollY + 120;
    var current = "";
    for (var i = sections.length - 1; i >= 0; i--) {
      var s = sections[i];
      if (s.offsetTop <= y) {
        current = s.id;
        break;
      }
    }
    links.forEach(function (link) {
      link.classList.toggle("is-active", link.getAttribute("href") === "#" + current);
    });
  }

  window.addEventListener("scroll", setActive, { passive: true });
  setActive();

  document.querySelectorAll(".tabs").forEach(function (root) {
    var buttons = root.querySelectorAll(".tabs__list button");
    var panels = root.querySelectorAll(".tab-panel");
    buttons.forEach(function (btn, idx) {
      btn.addEventListener("click", function () {
        var id = btn.getAttribute("data-tab");
        buttons.forEach(function (b) {
          b.setAttribute("aria-selected", b === btn ? "true" : "false");
        });
        panels.forEach(function (p) {
          p.classList.toggle("is-active", p.id === id);
        });
      });
      if (idx === 0) btn.setAttribute("aria-selected", "true");
    });
  });

  document.querySelectorAll(".copy-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var block = btn.closest(".code-block");
      var pre = block && block.querySelector("pre");
      if (!pre) return;
      var text = pre.textContent || "";
      navigator.clipboard.writeText(text).then(
        function () {
          var prev = btn.textContent;
          btn.textContent = "Copied";
          setTimeout(function () {
            btn.textContent = prev;
          }, 1500);
        },
        function () {
          btn.textContent = "Copy failed";
          setTimeout(function () {
            btn.textContent = "Copy";
          }, 1500);
        }
      );
    });
  });

  var LS_API = "g3_research_api_base";

  function servedFromFastApiUi() {
    try {
      var p = window.location.pathname || "";
      return p === "/ui" || p.startsWith("/ui/");
    } catch (e) {
      return false;
    }
  }

  /** Where POST /research lives: uvicorn (default :8000), not python http.server / Live Server. */
  function defaultApiBase() {
    try {
      if (window.location.protocol === "file:") {
        return "http://127.0.0.1:8000";
      }
      if (servedFromFastApiUi()) {
        return window.location.origin;
      }
      return "http://127.0.0.1:8000";
    } catch (e) {
      return "http://127.0.0.1:8000";
    }
  }

  var researchForm = document.getElementById("research-form");
  var apiBaseInput = document.getElementById("api-base");
  var questionInput = document.getElementById("research-question");
  var mockInput = document.getElementById("mock-tools");
  var submitBtn = document.getElementById("research-submit");
  var statusEl = document.getElementById("research-status");
  var errEl = document.getElementById("research-error");
  var outWrap = document.getElementById("research-output");
  var outAnswer = document.getElementById("out-answer");
  var outMeta = document.getElementById("out-meta");
  var outSubqs = document.getElementById("out-subqs");
  var outEvidenceBody = document.querySelector("#out-evidence tbody");
  var outJson = document.getElementById("out-json");
  var outSteps = document.getElementById("out-steps");
  var apiHealthEl = document.getElementById("api-health");
  var apiHealthRetry = document.getElementById("api-health-retry");
  var healthInputTimer = null;

  if (apiBaseInput) {
    var def = defaultApiBase();
    var stored = null;
    try {
      stored = window.localStorage.getItem(LS_API);
    } catch (e) {}

    var useDefault = !stored;
    if (stored && servedFromFastApiUi()) {
      try {
        var stNorm = stored.trim().replace(/\/$/, "");
        var storedOrig = new URL(stNorm + "/").origin;
        useDefault = storedOrig === window.location.origin;
      } catch (e4) {
        useDefault = true;
      }
    } else if (stored) {
      try {
        var normalized = stored.trim().replace(/\/$/, "");
        var storedOrigin = new URL(normalized + "/").origin;
        var pageOrigin = window.location.origin;
        if (!servedFromFastApiUi() && storedOrigin === pageOrigin) {
          useDefault = true;
        }
      } catch (e2) {
        useDefault = true;
      }
    }

    apiBaseInput.value = useDefault ? def : stored.trim().replace(/\/$/, "");
    if (useDefault) {
      try {
        window.localStorage.setItem(LS_API, apiBaseInput.value);
      } catch (e3) {}
    }
  }

  function setError(msg) {
    if (!errEl) return;
    if (!msg) {
      errEl.hidden = true;
      errEl.textContent = "";
      return;
    }
    errEl.hidden = false;
    errEl.textContent = msg;
  }

  function normalizeBase(raw) {
    raw = (raw || "").trim();
    if (!raw) raw = defaultApiBase();
    return raw.replace(/\/$/, "");
  }

  function setApiHealth(kind, text) {
    if (!apiHealthEl) return;
    apiHealthEl.className = "api-health api-health--" + kind;
    apiHealthEl.textContent = text;
  }

  function checkApiHealth() {
    if (!apiBaseInput) return;
    var base = normalizeBase(apiBaseInput.value);
    setApiHealth("pending", "Checking " + base + "/health …");
    fetch(base + "/health", { method: "GET", headers: { Accept: "application/json" } })
      .then(function (r) {
        if (!r.ok) throw new Error("bad status");
        return r.json();
      })
      .then(function (j) {
        var p = j && j.llm_provider != null ? String(j.llm_provider) : "—";
        setApiHealth("ok", "Connected — GET /health OK. llm_provider: " + p);
      })
      .catch(function () {
        setApiHealth(
          "bad",
          "No API at " + base + " (nothing listening or blocked). Start uvicorn, then Test connection again."
        );
      });
  }

  if (apiBaseInput && apiHealthEl) {
    apiBaseInput.addEventListener("input", function () {
      clearTimeout(healthInputTimer);
      healthInputTimer = setTimeout(checkApiHealth, 550);
    });
  }
  if (apiHealthRetry) {
    apiHealthRetry.addEventListener("click", checkApiHealth);
  }
  setTimeout(checkApiHealth, 400);

  function renderResult(data) {
    if (!outWrap || !outAnswer) return;
    outAnswer.textContent = data.answer || "";

    if (outMeta) {
      outMeta.innerHTML = "";
      var ex = data.execution || {};
      var ret = data.retrieval || {};
      var sess = data.session || {};
      var rows = [
        ["Elapsed (ms)", ex.elapsed_ms != null ? String(ex.elapsed_ms) : "—"],
        ["Injected context (est. tokens)", ret.selected_context_token_estimate != null ? String(ret.selected_context_token_estimate) : "—"],
        ["Memory store total (tokens)", ret.memory_store_token_total != null ? String(ret.memory_store_token_total) : "—"],
        ["Session prompt tokens", sess.prompt_tokens != null ? String(sess.prompt_tokens) : "—"],
        ["Session completion tokens", sess.completion_tokens != null ? String(sess.completion_tokens) : "—"],
        ["Est. cost (USD)", sess.estimated_cost_usd != null ? String(sess.estimated_cost_usd) : "—"],
      ];
      rows.forEach(function (r) {
        var dt = document.createElement("dt");
        dt.textContent = r[0];
        var dd = document.createElement("dd");
        dd.textContent = r[1];
        outMeta.appendChild(dt);
        outMeta.appendChild(dd);
      });
    }

    if (outSubqs) {
      outSubqs.innerHTML = "";
      (data.sub_questions || []).forEach(function (sq) {
        var li = document.createElement("li");
        var strong = document.createElement("strong");
        strong.textContent = "Q" + sq.id + ": ";
        li.appendChild(strong);
        li.appendChild(document.createTextNode(sq.question || ""));
        if (sq.focus) {
          li.appendChild(document.createElement("br"));
          var em = document.createElement("em");
          em.textContent = sq.focus;
          li.appendChild(em);
        }
        outSubqs.appendChild(li);
      });
      if (!data.sub_questions || !data.sub_questions.length) {
        var empty = document.createElement("li");
        empty.textContent = "—";
        outSubqs.appendChild(empty);
      }
    }

    if (outEvidenceBody) {
      outEvidenceBody.innerHTML = "";
      var ev = (data.retrieval && data.retrieval.selected_evidence) || [];
      if (!ev.length) {
        var tr = document.createElement("tr");
        var td = document.createElement("td");
        td.colSpan = 5;
        td.textContent = "—";
        tr.appendChild(td);
        outEvidenceBody.appendChild(tr);
      } else {
        ev.forEach(function (row) {
          var tr = document.createElement("tr");
          [
            row.chunk_id || "—",
            row.kind || "—",
            row.step != null ? String(row.step) : "—",
            row.token_estimate != null ? String(row.token_estimate) : "—",
            null,
          ].forEach(function (cell, i) {
            var td = document.createElement("td");
            if (i === 4) {
              if (row.url) {
                var a = document.createElement("a");
                a.href = row.url;
                a.textContent = "Open";
                a.rel = "noopener noreferrer";
                td.appendChild(a);
              } else {
                td.textContent = "—";
              }
            } else {
              td.textContent = cell;
            }
            tr.appendChild(td);
          });
          outEvidenceBody.appendChild(tr);
        });
      }
    }

    if (outJson) {
      outJson.textContent = JSON.stringify(data, null, 2);
    }
    if (outSteps) {
      outSteps.textContent = JSON.stringify((data.execution && data.execution.steps) || [], null, 2);
    }

    outWrap.hidden = false;
  }

  if (researchForm && submitBtn) {
    researchForm.addEventListener("submit", function (e) {
      e.preventDefault();
      setError("");
      if (outWrap) outWrap.hidden = true;

      var base = normalizeBase(apiBaseInput ? apiBaseInput.value : "");
      var q = (questionInput && questionInput.value) || "";
      var mock = mockInput ? mockInput.checked : true;

      try {
        window.localStorage.setItem(LS_API, base);
      } catch (ignore) {}

      submitBtn.disabled = true;
      if (statusEl) {
        statusEl.textContent = "Running research…";
        statusEl.classList.add("is-busy");
      }

      var url = base + "/research";
      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ question: q.trim(), mock_tools: mock }),
      })
        .then(function (res) {
          return res.text().then(function (text) {
            var parsed = null;
            try {
              parsed = text ? JSON.parse(text) : null;
            } catch (e) {
              parsed = null;
            }
            return { res: res, parsed: parsed, text: text };
          });
        })
        .then(function (_ref) {
          var res = _ref.res;
          var parsed = _ref.parsed;
          var rawText = _ref.text || "";
          if (!res.ok) {
            var detail =
              parsed && typeof parsed.detail === "string"
                ? parsed.detail
                : parsed && Array.isArray(parsed.detail)
                  ? parsed.detail.map(function (d) {
                      return d.msg || JSON.stringify(d);
                    }).join("; ")
                  : rawText || res.statusText;
            var looksLikeHtml = /^\s*</.test(rawText);
            if (
              res.status === 501 ||
              (looksLikeHtml && rawText.indexOf("Unsupported method") !== -1) ||
              (looksLikeHtml && rawText.indexOf("POST") !== -1 && res.status >= 400)
            ) {
              detail =
                "This API base URL is only serving static files (for example python -m http.server on port 8765). It cannot handle POST /research. Start the FastAPI app and point here to uvicorn instead, e.g. http://127.0.0.1:8000 — or open the bundled UI at http://127.0.0.1:8000/ui/";
            } else if (looksLikeHtml && String(detail).length > 200) {
              detail =
                res.status +
                " " +
                res.statusText +
                ": the server returned an HTML error page instead of JSON. Use the API base where uvicorn is running (default port 8000), not the port where you opened this HTML file.";
            }
            throw new Error(detail || "Request failed");
          }
          if (!parsed || typeof parsed !== "object") {
            throw new Error(
              "Expected JSON from /research. Check API base points to FastAPI (http://127.0.0.1:8000 by default)."
            );
          }
          renderResult(parsed);
        })
        .catch(function (err) {
          var msg = err && err.message ? err.message : String(err);
          if (
            msg === "Failed to fetch" ||
            msg.indexOf("NetworkError when attempting to fetch resource") !== -1
          ) {
            msg = [
              "The browser could not connect to " + base + " (nothing is accepting requests there).",
              "",
              "1) Open a terminal in your project folder (assignment_2).",
              "2) Activate your venv if you use one, then run:",
              "   uvicorn research_agent.api:app --host 127.0.0.1 --port 8000",
              "3) Wait until you see “Application startup complete”.",
              "4) Click “Test connection” above, then Run research again.",
              "",
              "Tip: opening http://127.0.0.1:8000/ui/ serves this page from the same process as the API."
            ].join("\n");
            checkApiHealth();
          }
          setError(msg);
        })
        .finally(function () {
          submitBtn.disabled = false;
          if (statusEl) {
            statusEl.textContent = "";
            statusEl.classList.remove("is-busy");
          }
        });
    });
  }
})();
