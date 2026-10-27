//langauge translate
  const translations = {
    en: {
      nav_home: "Home",
      nav_about: "About",
      hero_title: "Welcome to our website",
      hero_text: "Plan your studies smarter."
    },
    np: {
      nav_home: "गृहपृष्ठ",
      nav_about: "हाम्रो बारेमा",
      hero_title: "हाम्रो वेबसाइटमा स्वागत छ",
      hero_text: "आफ्नो अध्ययन अझ राम्रोसँग योजना गर्नुहोस्।"
    }
  };

  function setLanguage(lang) {
    const dict = translations[lang] || translations.en;

    document.querySelectorAll("[data-i18n]").forEach(el => {
      const key = el.getAttribute("data-i18n");
      if (dict[key] !== undefined) el.textContent = dict[key];
    });

    document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
      const key = el.getAttribute("data-i18n-placeholder");
      if (dict[key] !== undefined) el.placeholder = dict[key];
    });

    document.documentElement.lang = lang === "np" ? "ne" : "en";

    const radio = document.querySelector(`input[name="langOption"][value="${lang}"]`);
    if (radio) radio.checked = true;

    try { localStorage.setItem("lang", lang); } catch (e) {}
  }

  document.querySelectorAll('input[name="langOption"]').forEach(radio => {
    radio.addEventListener("change", e => setLanguage(e.target.value));
  });

  let saved = "en";
  try { saved = localStorage.getItem("lang") || "en"; } catch (e) {}
  setLanguage(saved);
