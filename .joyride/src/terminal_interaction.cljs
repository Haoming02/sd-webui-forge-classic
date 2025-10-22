(ns terminal.interaction
  (:require ["vscode" :as vscode]))

;; Helper function to find a terminal by name
(defn- find-terminal [name]
  (first (filter #(= (.-name %) name) (.-terminals vscode/window))))

;; Helper function to notify user
(defn- notify [message]
  (vscode/window.showInformationMessage message))

;; List all terminal names
(defn list-terminals []
  (map #(.-name %) (.-terminals vscode/window)))

;; Create a new terminal and notify
(defn create-terminal [name]
  (let [term (vscode/window.createTerminal name)]
    (notify (str "Created terminal: " name))
    term))

;; Send command to a terminal, creating it if it doesn't exist
(defn send-command [cmd & [term-name]]
  (let [term (if term-name
               (or (find-terminal term-name)
                   (let [new-term (create-terminal term-name)]
                     new-term))
               (or (first (.-terminals vscode/window))
                   (create-terminal "default")))]
    (when term
      (.sendText term cmd)
      (notify (str "Sent command: " cmd " to " (or term-name "default") " terminal")))))

;; Dispose a terminal by name and notify
(defn dispose-terminal [name]
  (let [term (find-terminal name)]
    (if term
      (do (.dispose term)
          (notify (str "Disposed terminal: " name)))
      (notify (str "Terminal not found: " name)))))

;; Get the process ID of a terminal (promise)
(defn get-terminal-pid [name]
  (when-let [term (find-terminal name)]
    (.-processId term)))

;; Get user input via input box (promise)
(defn get-user-input [prompt]
  (vscode/window.showInputBox {:prompt prompt}))

;; Select a terminal via quick pick (promise)
(defn select-terminal []
  (let [terms (list-terminals)]
    (vscode/window.showQuickPick (clj->js terms) {:placeHolder "Select a terminal"})))

;; Note: Output capture is event-based in VS Code API; use onDidWriteTerminalData for advanced cases