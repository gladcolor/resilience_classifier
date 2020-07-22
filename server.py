from waitress import serve
import main

serve(main.app, listen='*:8080')
