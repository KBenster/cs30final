function minIndex(theArray) {
    let indexOfSmallest = 0;
    for (let i = 0; i < theArray.length; i++) {
      if (theArray[i] < theArray[indexOfSmallest]) {
        indexOfSmallest = i;
      }
    }
    return indexOfSmallest;
  }
  
  function isSorted(theArray) {
    for (let i = 0; i < theArray.length; i++) {
      if (theArray[i] < theArray[i+1]) {
        return false;
      }
    }
    return true;
  }
  
  function selectionSort(theArray, iterations) {
    if (iterations === theArray.length || isSorted(theArray)) {
      return theArray;
    }
    console.log("new layer")
    let finalArray = theArray.slice(0, iterations); // a.push(...b)
    console.log("so far sorted " + finalArray)
    let indexOfSmallest = minIndex(theArray.slice(iterations));
    console.log("index of next smallest " + indexOfSmallest);
    finalArray.push(theArray[indexOfSmallest]);
    
    theArray.splice(iterations+1, 1)
    finalArray.push(...theArray);
    console.log(finalArray);
    selectionSort(finalArray, iterations + 1)
  }
  
  function setup() {
    createCanvas(400, 400);
    
    print(selectionSort([5, 15, 3, 8, 9, 1, 20, 7], 0))
  }
  
  function draw() {
    background(220);
  }