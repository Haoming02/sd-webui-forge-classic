(ns workspace-info
  (:require ["vscode" :as vscode]))

(defn get-workspace-folders []
  (map #(.-name %) (.-workspaceFolders vscode/workspace)))

(defn get-workspace-root []
  (when-let [folders (.-workspaceFolders vscode/workspace)]
    (.-fsPath (.-uri (first folders)))))