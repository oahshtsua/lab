import { useEffect } from "react";
import { useTranslation } from "react-i18next";

const languages = [
  { code: "ar-SA", name: "Arabic" },
  { code: "de-DE", name: "Deutsch" },
  { code: "en-US", name: "English" },
  { code: "nn-NO", name: "Norsk" },
  { code: "fr-FR", name: "French" },
];
function LanguageSelector() {
  const { i18n } = useTranslation();

  useEffect(() => {
    document.documentElement.dir = i18n.dir();
  }, [i18n.language]);

  return (
    <div className="btn-container">
      {languages.map((language) => (
        <button
          key={language.code}
          className={language.code === i18n.language ? "selected" : ""}
          id={language.code}
          onClick={() => i18n.changeLanguage(language.code)}
        >
          {language.name}
        </button>
      ))}
    </div>
  );
}

export default LanguageSelector;
