// Windows version resource: the tray balloon title, the Task Manager name and the
// file icon all come from here. Other targets have nothing to embed.
fn main() {
    println!("cargo:rerun-if-changed=../../assets/phonon.ico");
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../assets/phonon.ico")
            .set("ProductName", "Phonon")
            .set("FileDescription", "Phonon")
            .set("CompanyName", "Phonon")
            .set("LegalCopyright", "MIT");
        res.compile()
            .expect("phonon-win: compiling the Windows version resource");
    }
}
