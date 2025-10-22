(ns current-selection
  (:require ["vscode" :as vscode]))

(defn get-current-selection []
  (let [editor vscode/window.activeTextEditor]
    (when editor
      (.getText (.-document editor) (.-selection editor)))))