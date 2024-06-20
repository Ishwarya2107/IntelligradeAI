<?php
// Check if the form was submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Check if file was uploaded without errors
    if (isset($_FILES["pdfFile"]) && $_FILES["pdfFile"]["error"] == 0) {
        $uploadDir = "uploads/"; // Directory where uploaded files will be saved
        $uploadedFile = $uploadDir . basename($_FILES["pdfFile"]["name"]); // Path to save the uploaded file
        // Move the uploaded file to the specified directory
        if (move_uploaded_file($_FILES["pdfFile"]["tmp_name"], $uploadedFile)) {
            echo "The file ". basename($_FILES["pdfFile"]["name"]). " has been uploaded.";
        } else {
            echo "Sorry, there was an error uploading your file.";
        }
    } else {
        echo "No file uploaded or an error occurred.";
    }
}
?>
