// VeriTile architecture overview — turn the rendered Verso page into a slide
// deck. Verso emits each top-level `# Heading` as a `<section>` inside the
// page's `<article>`; we tag each one as a `.slide` and only show one at a
// time. Arrow / PageDown / Space advance, Arrow / PageUp / Backspace go back.
// Press `f` for fullscreen.

(function () {
  "use strict";

  function buildDeck() {
    const root = document.getElementById("deck-root");
    if (!root) return [];

    const article = root.querySelector("article");
    if (!article) return [];

    // Page title is the first <h1> directly under <article>; sections come
    // after, one per top-level "# Heading" in the source. Anything between
    // the <h1> and the first <section> (e.g. a tagline paragraph written
    // before the first `# Heading` in the Verso doc) also lives on the
    // title slide.
    const pageTitle = article.querySelector(":scope > h1");
    const sections  = Array.from(article.querySelectorAll(":scope > section"));

    const titleSlide = document.createElement("section");
    titleSlide.className = "slide title";
    if (pageTitle) {
      let cur = pageTitle;
      while (cur) {
        const next = cur.nextSibling;
        if (next && next.nodeType === 1 && next.tagName === "SECTION") break;
        titleSlide.appendChild(cur); // detaches from article
        cur = next;
      }
    }

    // Re-mount inside a 16:9 stage container so CSS can letterbox to ratio.
    const stage = document.createElement("div");
    stage.className = "deck-stage";
    stage.appendChild(titleSlide);
    for (const s of sections) {
      s.classList.add("slide");
      stage.appendChild(s);
    }
    article.innerHTML = "";
    article.appendChild(stage);

    return [titleSlide].concat(sections);
  }

  // Prism doesn't ship a Lean 4 grammar — register a small one so Prism can
  // tokenise the embedded DSL / theorem / proof blocks.
  function registerLeanGrammar() {
    if (!window.Prism) return;
    if (Prism.languages.lean) return;
    Prism.languages.lean = {
      "comment": [
        { pattern: /\/-[\s\S]*?-\//, greedy: true },
        { pattern: /--.*/,            greedy: true },
      ],
      "string":   { pattern: /"(?:\\.|[^"\\])*"/, greedy: true },
      "keyword":  /\b(?:theorem|lemma|example|def|abbrev|let|fun|λ|by|do|if|then|else|match|with|namespace|end|open|import|class|structure|inductive|instance|where|in|return|have|show|from|exact|apply|refine|simp|rfl|rw|rewrite|cases|induction|intro|intros|sorry|forall|exists|fun)\b/,
      "boolean":  /\b(?:True|False)\b/,
      "function": [
        { pattern: /\btl\.\w+/ },
        { pattern: /\b(?:Kernel|RegionName|Nat|Int|Real|ℝ|ℤ|ℕ|Fin|BlockState|TensorView|Op|Stmt|TileShape|TileDType)\b/, alias: "class-name" },
      ],
      "number":   /\b\d+(?:\.\d+)?\b/,
      "operator": /:=|=>|→|←|⟶|⟵|·|⊢|::?|->|<-|∀|∃|≠|≤|≥|⟂|×|·/,
      "punctuation": /[(){}\[\];,]/,
    };
  }

  function detectLang(txt) {
    if (/@triton\.jit|tl\.constexpr|^\s*def\s+\w+\([^)]*tl\./m.test(txt)) {
      return "python";
    }
    if (/\btheorem\b|\bdef\s+\w+[\s\S]*?:\s*Kernel\b|:=\s*triton\b|⟶|→/.test(txt)) {
      return "lean";
    }
    return null;
  }

  function applySyntaxHighlight() {
    registerLeanGrammar();
    const pres = document.querySelectorAll("main pre");
    pres.forEach((pre) => {
      const txt = pre.textContent;
      const lang = detectLang(txt);
      if (!lang) return;
      let code = pre.querySelector(":scope > code");
      if (!code) {
        code = document.createElement("code");
        code.textContent = txt;
        pre.innerHTML = "";
        pre.appendChild(code);
      }
      code.className = "language-" + lang;
      if (window.Prism && Prism.languages && Prism.languages[lang]) {
        Prism.highlightElement(code);
      }
    });
  }

  function init() {
    const slides = buildDeck();
    if (slides.length === 0) return;

    applySyntaxHighlight();

    // Switch into slide mode only once we know the DOM was rebuilt successfully.
    document.body.classList.add("deck-ready");

    let idx = 0;
    const counter = document.getElementById("counter");
    const prev    = document.getElementById("prev");
    const next    = document.getElementById("next");

    function show(i) {
      idx = Math.max(0, Math.min(slides.length - 1, i));
      slides.forEach((s, k) => s.classList.toggle("active", k === idx));
      if (counter) counter.textContent = (idx + 1) + " / " + slides.length;
      try {
        history.replaceState(null, "", "#s" + (idx + 1));
      } catch (_) { /* file:// may reject pushState */ }
    }

    function go(delta) { show(idx + delta); }

    document.addEventListener("keydown", (ev) => {
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      switch (ev.key) {
        case "ArrowRight":
        case "PageDown":
        case " ":
          ev.preventDefault(); go(1); break;
        case "ArrowLeft":
        case "PageUp":
        case "Backspace":
          ev.preventDefault(); go(-1); break;
        case "Home":
          ev.preventDefault(); show(0); break;
        case "End":
          ev.preventDefault(); show(slides.length - 1); break;
        case "f":
        case "F":
          ev.preventDefault();
          if (document.fullscreenElement) document.exitFullscreen();
          else document.documentElement.requestFullscreen();
          break;
      }
    });

    if (prev) prev.addEventListener("click", () => go(-1));
    if (next) next.addEventListener("click", () => go(1));

    // Resume from #sN if present.
    const m = /^#s(\d+)$/.exec(location.hash);
    show(m ? parseInt(m[1], 10) - 1 : 0);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
