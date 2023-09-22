const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const base64img = require('base64-img');
const {PythonShell} = require('python-shell');
const { exec } = require("child_process");
const app = express();
const PORT = 3000;

// exec("powershell.exe .\yolo-env\Scripts\Activate", (error, stdout, stderr) => {
//     if (error) {
//         console.log(`error: ${error.message}`);
//         return;
//     }
//     if (stderr) {
//         console.log(`stderr: ${stderr}`);
//         return;
//     }
   
// });
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
async function imageDetect() {
    let ret = [];
    let options = {
        mode: 'text',
        pythonPath: 'hydroscore-env/bin/python3',
        args: ['public/static/image.png'],
    }
    ret = PythonShell.run('object-detect.py', options).then(results=>{
        // results is an array consisting of messages collected during execution
        //console.log(results);
        ret = [...results];
        // console.log(ret)
        return ret;
      });
    return ret;
      
    
}

app.use(cors({
    origins: '*',
}));
app.use(bodyParser.json({limit: '50mb', extended: true}));
app.use(bodyParser.urlencoded({limit: "50mb", extended: true, parameterLimit:50000}));
app.use(bodyParser.text({ limit: '200mb' }));
app.use(express.json());
app.get('/', (req, res) => {
    res.send('<h1>Welcome to HydroScore backend</h1>');
    res.status(200);
});

app.post('/upload', async (req, res) => {
    const { image } = req.body;
    // console.log(image);
    base64img.img(image, './public/static', 'image', function(err, filepath) {
        const pathArr = filepath.split('/');
        const fileName = pathArr[pathArr.length - 1];
        // res.status(200).json({
        //     success: true,
        //     url: `http://127.0.0.1:${PORT}/${fileName}`,
        // });
        
    });
    s = await imageDetect()
    res.send(s);
   
});

app.listen(PORT, () => {
    console.log(`App is listening at http://localhost:${PORT}`);
});
