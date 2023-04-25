const http = require('http');

const hostname = '127.0.0.1';
const port = 3000;

const server = http.createServer((req, res) => {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'text/plain');
  res.end('Hello World');
});

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});

//https://www.sohamkamani.com/nodejs/session-cookie-authentication/
//https://mariadb.com/kb/en/getting-started-with-the-nodejs-connector/
//https://github.com/sohamkamani/nodejs-session-cookie-example