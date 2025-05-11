import "./App.css";
import { Trans, useTranslation } from "react-i18next";
import LanguageSelector from "./components/language-selector";

function App() {
  useTranslation();
  return (
    <div className="container">
      <LanguageSelector />
      <span>
        <Trans
          i18nKey={"greeting"}
          values={{
            name: "Cookie",
          }}
          components={{ 1: <b /> }}
        />
      </span>
    </div>
  );
}

export default App;
