function redraw(){
  var figs = document.getElementsByClassName("js-plotly-plot")
  for (var i = 0; i < figs.length; i++) {
    console.log("redrawing fig: " + i)
    Plotly.redraw(figs[i])
  }
}

console.log("Start");
setTimeout(function(){
    console.log("Redraw");
    redraw();
}, 2000);
console.log("End");