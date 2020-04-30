import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import {Subscription} from 'rxjs';
import { BackendconnectionService } from '../backendconnection.service';
import { FormBuilder, Validators, FormGroup  } from '@angular/forms';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage {
  Subs: Subscription;
  fileToUpload: File = null;
  isSubmitted = false;
  myForm: FormGroup;
  process = false;
  isdatavailable = null;
  streamdata = null;


  constructor(private backendApi: BackendconnectionService, private http: HttpClient, private formBuilder: FormBuilder) {
    this.myForm = formBuilder.group({
      imageurl: ['', Validators.required]
    });
  }



    // tslint:disable-next-line:use-lifecycle-interface
    ngOnInit() {
    }

    // tslint:disable-next-line:use-lifecycle-interface
    ngOnDestroy() {
      this.Subs.unsubscribe();
    }

    submit() {

      this.isSubmitted = true;
      if (this.myForm.invalid) {
        return;
      }
      this.process = true;
      console.log(this.myForm.value);
     // post to api

      if (this.fileToUpload != null) {

        const imageurl = this.fileToUpload;
        this. Subs = this.backendApi
        .postPredictor(imageurl)
        .subscribe( data => {
          if (data != null) {

            const reader = new FileReader();
            reader.readAsDataURL(this.fileToUpload); // toBase64
            reader.onload = () => {
                    this.isdatavailable = reader.result as string; // base64 Image src
                  };

            this.process = false;
            this.streamdata = data;
          }
          console.log(data);
        });
     }


    }

    handleFileInput(files: FileList) {
      this.fileToUpload = files.item(0);
  }

  reset() {
    this.process = false;
    this.streamdata = null;
    this.isdatavailable = null;
    this.fileToUpload = null;
  }

}
