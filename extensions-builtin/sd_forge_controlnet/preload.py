def preload(parser):
    parser.add_argument(
        "--controlnet-loglevel",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
