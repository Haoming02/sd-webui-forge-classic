(ns open-editors
  (:require ["vscode" :as vscode]))

(defn get-open-editors []
  (map #(.-fileName (.-document %)) vscode/window.visibleTextEditors))