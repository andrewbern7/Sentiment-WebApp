const express = require('express');
const path    = require('path');
const app     = express();
const PORT    = process.env.PORT || 3000;

// parse JSON bodies if you have API endpoints:
app.use(express.json());

// serve everything in /public at the web root
app.use(express.static(path.join(__dirname, 'public')));

// fallback for client-side routing (optional)
// app.get('*', (req, res) => res.sendFile(path.join(__dirname, 'public/index.html')));

app.listen(PORT, () => {
  console.log(`Server listening at http://localhost:${PORT}`);
});
