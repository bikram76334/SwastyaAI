document.addEventListener("DOMContentLoaded", () => {
  const CHUNK_SIZE = 100; // your Flask route accepts max 128 strings per request

  const SKIP_SELECTOR =
    'script, style, noscript, textarea, code, pre, .btn-group[aria-label="Language"], [data-no-translate], .notranslate';

  const ATTRS = ["placeholder", "title", "alt"];
  const ATTR_SELECTOR = ATTRS.map((a) => `[${a}]`).join(",");

  // ---------- Devanagari digits ----------
  const DEVANAGARI = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"];

  // alone (so B0, B1, E1 stay as they are).
  function toDevanagari(str) {
    return str.replace(/[0-9]/g, (d, offset, whole) => {
      const prev = whole[offset - 1] || "";
      if (/[A-Za-z]/.test(prev)) return d;

      let i = offset - 1;
      while (i >= 0 && /[0-9]/.test(whole[i])) i--;
      if (i >= 0 && /[A-Za-z]/.test(whole[i]) && i < offset - 1) return d;
      return DEVANAGARI[d];
    });
  }

  const originals = new WeakMap(); 
  const lastSet = new WeakMap();   
  let currentLang = "en";
  let currentRequest = 0;

  // ---------- collecting things to translate ----------
  function skip(el) {
    return !el || el.closest(SKIP_SELECTOR);
  }

  function addTextNode(node, out) {
    if (node.nodeValue.trim() && !skip(node.parentElement)) {
      out.push({ node, attr: null });
    }
  }

  function addAttrs(el, out) {
    if (skip(el)) return;
    ATTRS.forEach((a) => {
      if (el.hasAttribute(a) && el.getAttribute(a).trim()) {
        out.push({ node: el, attr: a });
      }
    });
  }

  function collect(root, out = []) {
    if (root.nodeType === 3) {
      addTextNode(root, out);
      return out;
    }
    if (root.nodeType !== 1 || skip(root)) return out;

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) addTextNode(walker.currentNode, out);

    addAttrs(root, out);
    root.querySelectorAll(ATTR_SELECTOR).forEach((el) => addAttrs(el, out));
    return out;
  }

  // ---------- original English / set value ----------
  function getOriginal(t) {
    if (t.attr) {
      const key = "en" + t.attr;
      if (!(key in t.node.dataset)) {
        t.node.dataset[key] = t.node.getAttribute(t.attr);
      }
      return t.node.dataset[key];
    }
    if (!originals.has(t.node)) originals.set(t.node, t.node.nodeValue);
    return originals.get(t.node);
  }

  function setValue(t, value) {
    if (t.attr) {
      t.node.setAttribute(t.attr, value);
    } else {
      lastSet.set(t.node, value);
      t.node.nodeValue = value;
    }
  }

  // ---------- talk to your Flask  route ----------
  async function translateBatch(texts) {
    const res = await fetch("/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts, target: "ne" }),
    });
    if (!res.ok) throw new Error("Translate failed: " + res.status);
    const data = await res.json();
    return data.translations;
  }

  async function translateTargets(targets, requestId) {
    const english = targets.map(getOriginal);
    const unique = [...new Set(english.map((s) => s.trim()))];
    const map = {};

    for (let i = 0; i < unique.length; i += CHUNK_SIZE) {
      const chunk = unique.slice(i, i + CHUNK_SIZE);
      const result = await translateBatch(chunk);
      chunk.forEach((text, j) => (map[text] = result[j]));
    }

    if (requestId !== currentRequest) return; // user toggled again

    targets.forEach((t, i) => {
      const nepali = map[english[i].trim()] || english[i];
      setValue(t, toDevanagari(nepali));
    });
  }

  function restoreEnglish(targets) {
    targets.forEach((t) => setValue(t, getOriginal(t)));
  }

  // ---------- switch language ----------
  async function setLanguage(lang) {
    currentLang = lang;
    const id = ++currentRequest;
    document.documentElement.lang = lang === "np" ? "ne" : "en";

    const targets = collect(document.body);

    try {
      if (lang === "np") {
        await translateTargets(targets, id);
      } else {
        restoreEnglish(targets);
      }
    } catch (err) {
      console.error(err);
    }

    const radio = document.querySelector(`input[name="langOption"][value="${lang}"]`);
    if (radio) radio.checked = true;

    try {
      localStorage.setItem("lang", lang);
    } catch (e) {}
  }

  // ---------- translate content added later  ----------
  const pending = new Set();
  let timer = null;

  async function flush() {
    const id = currentRequest;
    const targets = [];

    pending.forEach((n) => {
      if (document.body.contains(n)) collect(n, targets);
    });
    pending.clear();

    if (!targets.length) return;

    try {
      await translateTargets(targets, id);
    } catch (err) {
      console.error(err);
    }
  }

  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.type === "characterData") {
        const n = m.target;
        if (lastSet.get(n) === n.nodeValue) continue; // our own change
        originals.delete(n); // page wrote new English text
        pending.add(n);
      } else if (m.type === "childList") {
        m.addedNodes.forEach((n) => pending.add(n));
      }
    }

    if (currentLang !== "np") {
      pending.clear();
      return;
    }

    clearTimeout(timer);
    timer = setTimeout(flush, 100);
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true,
  });

  // ---------- toggle listeners ----------
  document.querySelectorAll('input[name="langOption"]').forEach((radio) => {
    radio.addEventListener("change", (e) => setLanguage(e.target.value));
  });

  // ---------- restore saved language ----------
  let saved = "en";
  try {
    saved = localStorage.getItem("lang") || "en";
  } catch (e) {}

  if (saved !== "en") setLanguage(saved);
});